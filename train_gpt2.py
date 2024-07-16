import os
import math
import time
import torch
from grokfast import gradfilter_ema
from gpt2 import GPTConfig, GPT
from dataloader import DataLoaderLite
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from torch.nn import functional as F

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# pytorch can be serious about its device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if master_process:
    wandb.init(project="ChirpGPT")

# sequence length
T = 1024
total_batch_size = T*32  # pick something about 32768
# micro batch size
B = 16


assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", ddp=ddp, master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", ddp=ddp, master_process=master_process)

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(), init_weights=True)
grads = None

#original_state_dict = torch.load('./BIRD_32khz_Small_model_41764.pt', map_location=torch.device('cpu'))

# Corrected state dictionary
#state_dict = {
#    'model': OrderedDict([
#        (key.replace('_orig_mod.', ''), value) for key, value in original_state_dict['model'].items()
#    ]),
#    'config': original_state_dict['config'],
#    'step': original_state_dict['step'],
#    'val_loss': original_state_dict['val_loss']
#}

#model.load_state_dict(state_dict['model'])

model.to(device)
use_compile = True
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

GROK_ALPHA = 0.9
GROK_LAMB = 0.625
weight_decay = 0.13333

max_lr = 4.75e-4
init_lr_pct = 0.06667
min_lr_pct = 0.025
neg_loss_weight = 0.6667

num_epochs = 25
grad_clip_percentile = 0.1
grad_clip_min = 1e-3
grad_clip_max = 0.65
norms_window_size = 300
# Decrease lr when norm percentile & loss are increasing
max_clip_slope = 1.085
lr_adj_rate = 0.925  # effectively, max_lr = max_lr * lr_adj_rate every norms_window_size/3 steps while conditions met
# Was 16000 for 24khz model, ~21000 for 32khz. Our goal is 610k +  (~2TB of 32khz audio if it were single epoch, or ~57.5GB tokenized)
max_steps = (num_epochs * train_loader.total_tokens) // total_batch_size
warmup_steps = int(max_steps*0.07)
print(f"Max steps: {max_steps}, Warmup steps: {warmup_steps}")


def get_lr(it):
    min_lr = max_lr * min_lr_pct
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (init_lr_pct + (1.0 - init_lr_pct) * (float(it) / float(max(1, warmup_steps))))
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def get_clip_value(norms_window, step):
    global max_lr, total_panic, optimizer, optimizer_resets
    if step < max(norms_window_size, warmup_steps * 0.5):
        return 1.0
    else:
        clip_value = np.percentile(norms_window, grad_clip_percentile * 100)

        # Decrease lr when norm percentile & loss are increasing
        third = norms_window_size // 3
        p1 = np.percentile(norms_window[:third], grad_clip_percentile * 100)
        p2 = np.percentile(norms_window[third:2*third], grad_clip_percentile * 100)
        p3 = np.percentile(norms_window[2*third:], grad_clip_percentile * 100)
        l1 = np.mean(loss_window[:third])
        l2 = np.mean(loss_window[third:2*third])
        l3 = np.mean(loss_window[2*third:])
        if step > warmup_steps * 0.5 and p3 > p2 > p1 and p3 / p2 > max_clip_slope and p2/p1 > max_clip_slope and l3 > l2 > l1:
            max_lr *= lr_adj_rate ** (3 / norms_window_size)
            total_panic += 1
            if total_panic % (third * 2) == 0:
                print("Too much panic: Resetting optimizer.")
                optimizer = raw_model.configure_optimizers(weight_decay=weight_decay,
                                                           learning_rate=get_lr(step), device_type=device_type,
                                                           log=master_process)
                optimizer_resets += 1
            wandb.log({
                "debug/panic": 1.0,
                "debug/total_panic": total_panic,
                "debug/max_lr": max_lr,
                "debug/optimizer_resets": optimizer_resets,
            }, step=step)
        else:
            wandb.log({
                "debug/panic": 0.0,
                "debug/total_panic": total_panic,
                "debug/max_lr": max_lr,
                "debug/optimizer_resets": optimizer_resets,
            }, step=step)

        return max(grad_clip_min, min(grad_clip_max, clip_value))


norms_window = []
loss_window = []
current_epoch = 0
total_panic = 0
optimizer_resets = 0
clip_val = get_clip_value([], 0)
optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr * init_lr_pct, device_type=device_type, log=master_process)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
#log_file = os.path.join(log_dir, f"log.txt")
#with open(log_file, "w") as f: # open for writing to clear the file
 #   pass

for step in tqdm(range(max_steps), f"Training..."):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 100 == 0 or last_step:
        model.eval()
        #val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 15
            for _ in range(val_loss_steps):
                x1, x2, y, _, _ = val_loader.next_batch()
                x = torch.cat([x1, x2], dim=1).to(device)
                y = y.to(device)
                if 'cuda' in device:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, _ = model(x)
                        loss = F.cross_entropy(logits[:, -512:].contiguous().view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits[:, -512:].contiguous().view(-1, logits.size(-1)), y.view(-1))

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            #print(f"validation loss: {val_loss_accum.item():.4f}")
            #with open(log_file, "a") as f:
            #    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            wandb.log({
                "val/loss": val_loss_accum.item()
            }, step=step)
            if step > 0 and ((step % 1000 == 0) or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                print(f"writing checkpoint to {checkpoint_path}")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum_pos = 0.0
    loss_accum_neg = 0.0
    loss_accum_combined = 0.0

    for micro_step in range(grad_accum_steps):
        x1_pos, x2_pos, y_pos, x2_neg, y_neg = train_loader.next_batch()

        x_pos = torch.cat([x1_pos, x2_pos], dim=1).to(device)
        y_pos = y_pos.to(device)
        x_neg = torch.cat([x1_pos, x2_neg], dim=1).to(device)
        y_neg = y_neg.to(device)

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits_pos, _ = model(x_pos)
            logits_neg, _ = model(x_neg)

            # Calculate loss only for the second chunk
            loss_pos = F.cross_entropy(logits_pos[:, -512:].contiguous().view(-1, logits_pos.size(-1)), y_pos.view(-1))
            loss_neg = F.cross_entropy(logits_neg[:, -512:].contiguous().view(-1, logits_neg.size(-1)), y_neg.view(-1))

            # Combine losses
            warmup_factor = min(step / (2.0 * warmup_steps), 1.0)
            # Clamp loss_neg so the division doesn't get nasty
            clamped_loss_neg = torch.clamp(loss_neg, min=0.75 * loss_pos, max=1.3333 * loss_pos)
            # Gradually introduce the negative loss
            #loss = loss_pos / (1.0 + warmup_factor * (clamped_loss_neg - 1.0) * neg_loss_weight + 1e-8)
            loss = loss_pos + ((loss_pos / clamped_loss_neg) * warmup_factor * neg_loss_weight)

        loss_accum_pos += loss_pos.detach() / grad_accum_steps
        loss_accum_neg += loss_neg.detach() / grad_accum_steps
        loss_accum_combined += loss.detach() / grad_accum_steps
        loss = loss / grad_accum_steps

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

        if step >= (warmup_steps * 0.6667):
            grads = gradfilter_ema(model, grads=grads, alpha=GROK_ALPHA,
                                   lamb=GROK_LAMB * (min(1.0, step / (warmup_steps * 1.5)) ** 3))

    if ddp:
        dist.all_reduce(loss_accum_pos, op=dist.ReduceOp.AVG)
        dist.all_reduce(loss_accum_neg, op=dist.ReduceOp.AVG)
        dist.all_reduce(loss_accum_combined, op=dist.ReduceOp.AVG)
    clip_val = get_clip_value(norms_window, step)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    norms_window.append(norm.item())
    loss_window.append(loss_accum_combined.item())
    if len(norms_window) > norms_window_size:
        norms_window.pop(0)
        loss_window.pop(0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_per_sec = total_batch_size / dt
    if master_process:
        #print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        #with open(log_file, "a") as f:
        #    f.write(f"{step} train {loss_accum.item():.6f}\n")
        prev_epoch = current_epoch
        current_epoch = step * total_batch_size // train_loader.total_tokens
        if prev_epoch != current_epoch:
            print(f"Epoch {current_epoch}")
        wandb.log({
            "etc/step": step,
            "etc/epoch": current_epoch,
            "etc/lr": lr,
            "etc/norm": norm.item(),
            "etc/clip_value": clip_val,
            "etc/toks_per_sec": tokens_per_sec,
            "train/loss_combined": loss_accum_combined.item(),
            "train/loss_pos": loss_accum_pos.item(),
            "train/loss_neg": loss_accum_neg.item(),
        }, step=step)

if ddp:
    destroy_process_group()
