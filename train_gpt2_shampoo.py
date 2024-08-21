import os
import math
import time
import torch
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
import random
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, DDPShampooConfig, CommunicationDType

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

# ------------------------------
# HYPER-PARAMETERS
# ------------------------------

# sequence length
T = 3072
total_batch_size = T*40  # pick something about 32768
# micro batch size
B = 8

grok_alpha = 0.925
grok_lamb = 1.1
weight_decay = 0.113333

# pi is just being silly, ~3.15 was arrived at experimentally
max_lr = math.pi * 1e-4
init_lr_pct = 0.075
min_lr_pct = 0.01

loss_by_later_subchunks = False
# When loss_by_later_subchunks = True, warmup to:
third_subchunk_predict_percentage = 0.75
# After warmup, 2nd+third subchunk percentage = 1 - third (during warmup full sequence likelihood decreases from 1 to 0)

# At 170MB tokenized data & next-chunk loss, val starts increasing ~epoch 5-6. With music at least seems start earlier for full sequence loss.
# Maybe try 3-4 epochs full-sequence then ~2-4(?) next-chunk(s)
num_epochs = 0.33 #3  # Can be fraction
warmup_steps = 2600  # was 2200

# Shampoo
max_preconditioner_dim = 4096
precondition_frequency = 100
start_preconditioning_step = warmup_steps // 3

resume = False
resume_from = './BIRD_FINAL_32khz_Small_NoTest_model_64432.pt'

# ------------------------------
# END HYPER-PARAMETERS
# ------------------------------

chunk_size = T // 3
print(f"Chunk size: {chunk_size}")

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", ddp=ddp, master_process=master_process, critical_divisor=chunk_size)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", ddp=ddp, master_process=master_process, critical_divisor=chunk_size)

# Was 15-64k. In theory our goal is 610k +  (~2TB of 32khz audio if it were single epoch, or ~57.5GB tokenized [total, obviously some # epochs > 1 is fine, at least 6 & probably more w/ more data])
max_steps = (num_epochs * train_loader.total_tokens) // total_batch_size
print(f"Max steps: {max_steps}, Warmup steps: {warmup_steps}")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(block_size=T), init_weights=True)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=max_lr * init_lr_pct,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=weight_decay,
    max_preconditioner_dim=max_preconditioner_dim,
    precondition_frequency=precondition_frequency,
    start_preconditioning_step=start_preconditioning_step,
    use_decoupled_weight_decay=True,
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32,
        num_trainers_per_group=8,
        communicate_params=False,
    ) if ddp else None,
)

if resume:
    checkpoint_path = resume_from
    print(f"Resuming from checkpoint: {checkpoint_path}")

    state_dict = {}
    torch.distributed.checkpoint.load_state_dict(
        state_dict=state_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(checkpoint_path),
    )

    # Load model state
    model_state_dict = OrderedDict([
        (key.replace('_orig_mod.', ''), value) for key, value in state_dict['model'].items()
    ])
    model.load_state_dict(model_state_dict)

    # Load optimizer state
    optimizer.load_distributed_state_dict(state_dict["optim"], key_to_param=model.named_parameters())

    if "step" in state_dict:
        step = state_dict["step"]
    if "val_loss" in state_dict:
        best_val_loss = state_dict["val_loss"]

    print("Checkpoint loaded successfully")

model.to(device)
use_compile = True
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


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


# Returns the likelihood of calculating loss by full sequence
def get_loss_likelihood(step):
    if not loss_by_later_subchunks:
        return 1.0
    else:
        if step > warmup_steps:
            return 0.0
        else:

            return 1.0 - (float(step) / float(max(1, warmup_steps)))


best_val_loss = 999.9
eval_every = 400
val_loss_steps = 10
current_epoch = 0

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
#log_file = os.path.join(log_dir, f"log.txt")
#with open(log_file, "w") as f: # open for writing to clear the file
 #   pass

t = tqdm(range(max_steps), f"Training epoch 1 of {num_epochs}")
for step in t:
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step > 0 and step % eval_every == 0 or last_step:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(val_loss_steps):
                r = random.random()
                # Full sequence loss
                if r < get_loss_likelihood(step):
                    x, y = val_loader.next_batch(False)
                    x = x.to(device)
                    y = y.to(device)
                    if 'cuda' in device:
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)
                # 3rd or 2nd+3rd subchunk loss
                else:
                    x, y, z = val_loader.next_batch(True)
                    x = x.to(device)
                    y = y.to(device)
                    z = z.to(device)
                    if 'cuda' in device:
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, _ = model(x)
                    else:
                        logits, _ = model(x)
                    # Predict third subchunk
                    if r < third_subchunk_predict_percentage:
                        inputs = logits[:, -chunk_size:].contiguous().view(-1, logits.size(-1))
                        targets = z.view(-1)
                    # Predict second+third subchunks
                    else:
                        inputs = logits[:, -chunk_size*2:].contiguous().view(-1, logits.size(-1))
                        targets = torch.cat([y, z], dim=1).view(-1)
                    loss = F.cross_entropy(inputs, targets)

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
            if last_step or (step > warmup_steps and val_loss_accum.item() < best_val_loss):
                best_val_loss = min(best_val_loss, val_loss_accum.item())
                if best_val_loss < 5.225:  #4.75:
                    val_loss_steps = 70
                    eval_every = 80
                elif best_val_loss < 5.365:  #4.825:
                    val_loss_steps = 50
                    eval_every = 120
                else:
                    val_loss_steps = 10
                    eval_every = 400
            #if step > 0 and (step % 1600 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_s{step:05d}_vl{val_loss_accum.item():.4f}")
                print(f"writing checkpoint to {checkpoint_path}")
                state_dict = {
                    "model": model.state_dict(),
                    "optim": optimizer.distributed_state_dict(key_to_param=model.named_parameters()),
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.distributed.checkpoint.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=torch.distributed.checkpoint.FileSystemWriter(checkpoint_path),
                )

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        r = random.random()
        # Full sequence loss
        if r < get_loss_likelihood(step):
            x, y = train_loader.next_batch(False)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss = model(x, y)
        # 3rd or 2nd+3rd subchunk loss
        else:
            x, y, z = train_loader.next_batch(True)
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(x)
            # Predict third subchunk
            if r < third_subchunk_predict_percentage:
                inputs = logits[:, -chunk_size:].contiguous().view(-1, logits.size(-1))
                targets = z.view(-1)
            # Predict second+third subchunks
            else:
                inputs = logits[:, -chunk_size * 2:].contiguous().view(-1, logits.size(-1))
                targets = torch.cat([y, z], dim=1).view(-1)
            loss = F.cross_entropy(inputs, targets)

        loss_accum += loss.detach() / grad_accum_steps
        loss = loss / grad_accum_steps

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

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
            #print(f"Epoch {current_epoch}")
            t.set_description(f"Training epoch {current_epoch+1} of {num_epochs}", refresh=True)
        wandb.log({
            "etc/step": step,
            "etc/epoch": current_epoch,
            "etc/lr": lr,
            "etc/toks_per_sec": tokens_per_sec,
            "train/loss": loss_accum.item(),
        }, step=step)

if ddp:
    destroy_process_group()
