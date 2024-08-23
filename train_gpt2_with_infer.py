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
import random
from two_sep_tokenizer import AudioTokenizer
from scipy.io.wavfile import write
from traceback import format_exc

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
    wandb.init(project="MusicGPT")

# ------------------------------
# HYPER-PARAMETERS
# ------------------------------

# sequence length
T = 3072
total_batch_size = T*42  # pick something about 32768
# micro batch size
B = 3

grok_enabled = True
grok_start_divergence = 1.075
divergence_window_size = 5
grok_warmup_steps = 1000
grok_alpha = 0.75  # 0.925
grok_lamb = 0.9  # 1.1

weight_decay = 0.113333

# Big music model is ~2.79x larger than Small
max_lr = 1.73e-4  #2.825e-4
init_lr_pct = 0.07
min_lr_pct = 0.01

loss_by_later_subchunks = False
# When loss_by_later_subchunks = True, warmup to:
third_subchunk_predict_percentage = 0.75
# After warmup, 2nd+third subchunk percentage = 1 - third (during warmup full sequence likelihood decreases from 1 to 0)

# At 170MB tokenized data & next-chunk loss, val starts increasing ~epoch 5-6. With music at least seems start earlier for full sequence loss.
# Maybe try 3-4 epochs full-sequence then ~2-4(?) next-chunk(s)
num_epochs = 3  # Can be fraction
grad_clip_percentile = 0.09
grad_clip_min = 1e-3
grad_clip_max = 0.85
norms_window_size = 250
# Decrease lr when norm percentile & loss are increasing
max_clip_slope = 1.1
lr_adj_rate = 0.925  # effectively, max_lr = max_lr * lr_adj_rate every norms_window_size/3 steps while conditions met
warmup_steps = 1800
save_every = 2500
inference_batch_size = 3

resume = True
resume_from = './log/model_2400.pt'
# Whether to reset (or load from checkpoint) the optimizer. Also resets norms&loss windows.
reset_optimizer = False
# Whether to reset (or load from checkpoint) the schedule (currently, the step number and best val loss)
reset_schedule = False

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
max_steps = int((num_epochs * train_loader.total_tokens) // total_batch_size)
print(f"Max steps: {max_steps}, Warmup steps: {warmup_steps}")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(block_size=T), init_weights=True)
model.to(device)
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr * init_lr_pct, device_type=device_type, log=False)
grads = None
step = 0
best_val_loss = 999.9
norms_window = []
loss_window = []
divergence_window = []
grok_start_step = -1

if resume:
    print(f"Resuming from {resume_from}")
    chkpt = torch.load(resume_from, map_location=torch.device('cpu'))
    model.load_state_dict(OrderedDict([
            (key.replace('_orig_mod.', ''), value) for key, value in chkpt['model'].items()
        ]))
    if not reset_optimizer:
        optimizer.load_state_dict(chkpt['optim'])
        if "norms_window" in chkpt:
            norms_window = chkpt["norms_window"]
        if "loss_window" in chkpt:
            loss_window = chkpt["loss_window"]
        if "divergence_window" in chkpt:
            divergence_window = chkpt["divergence_window"]
        if "grok_start_step" in chkpt:
            grok_start_step = chkpt["grok_start_step"]
    if not reset_schedule:
        if "step" in chkpt:
            step = chkpt["step"]
        if "val_loss" in chkpt:
            best_val_loss = chkpt["val_loss"]


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


def get_clip_value(norms_window, step):
    global max_lr, total_panic, optimizer, optimizer_resets
    if step < warmup_steps * 0.85 or len(norms_window) < norms_window_size:
        return grad_clip_max
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
        #print(f"p3: {p3}, p2: {p2}, p1: {p1}, l3: {l3}, l2: {l2}, l1: {l1}")
        #print(f"p3 slope {p3 / p2}, p2 slope {p2 / p1}")
        if p3 > p2 > p1 and p3 / p2 > max_clip_slope and p2 / p1 > max_clip_slope and l3 > l2 > l1:
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


# Returns the likelihood of calculating loss by full sequence
def get_loss_likelihood(step):
    if not loss_by_later_subchunks:
        return 1.0
    else:
        if step > warmup_steps:
            return 0.0
        else:
            return 1.0 - (float(step) / float(max(1, warmup_steps)))


def generate_tokens(model, seq_length, batch_size, prefill=None, temperature=0.95, top_k=650):
    if prefill is None:
        tokens = [4097]
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(batch_size, 1)
        x = tokens.to(device)
    else:
        x = prefill

    with torch.no_grad():
        for _ in tqdm(range(T - x.size(1) + 1), dynamic_ncols=True, desc="Generating tokens", position=1, leave=False):
            logits, _ = model(x)
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature
            # Handle NaN and Inf values in logits
            nan_mask = torch.isnan(next_token_logits) | torch.isinf(next_token_logits)
            if nan_mask.any():
                # print("Warning: NaN or Inf values detected in logits. Replacing with very negative values.")
                next_token_logits = torch.where(nan_mask, torch.full_like(next_token_logits, -1e9), next_token_logits)
            # Compute softmax probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            # Perform top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            # Renormalize the top-k probabilities
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            # Check for NaN values in top_k_probs
            if torch.isnan(top_k_probs).any():
                # print("Warning: NaN values detected in top-k probabilities. Using uniform distribution.")
                top_k_probs = torch.ones_like(top_k_probs) / top_k
            # Sample from the top-k distribution
            try:
                sample_indices = torch.multinomial(top_k_probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, sample_indices)
            except RuntimeError as e:
                print(f"Error during sampling: {e}")
                print("Falling back to argmax selection from top-k.")
                next_token = top_k_indices[:, 0].unsqueeze(-1)  # Select the highest probability token
            # Append the new token to the sequence
            x = torch.cat([x, next_token], dim=1)

    output_tokens = []
    for i in range(batch_size):
        tokens = x[i, -(seq_length+1):].tolist()
        output_tokens.append(tokens)
    return output_tokens


def save_audio_files(tokens, tokenizer, folder, prefix):
    for i in tqdm(range(len(tokens)), dynamic_ncols=True, desc="Decoding audio and saving files", position=0, leave=True):
        audio = tokenizer.decode(np.array([tokens[i]]))
        audio_np = audio.cpu().detach().numpy()
        # Normalize to [-1, 1] range
        audio_np = audio_np / np.max(np.abs(audio_np))
        # Convert to 16-bit PCM
        audio_16bit = (audio_np * 32767).astype(np.int16)
        filename = os.path.join(folder, f"{prefix}_{i}.mp3")
        write(filename, tokenizer.sample_rate, audio_16bit)
    del audio


eval_every = 50  # Gets changed below
val_loss_steps = 10  # Gets changed below
current_epoch = step * total_batch_size // train_loader.total_tokens
total_panic = 0
optimizer_resets = 0
clip_val = get_clip_value([], 0)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
#log_file = os.path.join(log_dir, f"log.txt")
#with open(log_file, "w") as f: # open for writing to clear the file
 #   pass

t = tqdm(range(step, max_steps), initial=step, total=max_steps, desc=f"Training epoch {current_epoch+1} of {num_epochs}", dynamic_ncols=True)
for step in t:
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step > 0 and (step % eval_every == 0 or step % save_every == 0 or last_step):
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
            divergence_window.append(val_loss_accum.item() / loss_window[-1])
            if len(divergence_window) > divergence_window_size:
                divergence_window.pop(0)

            if step == eval_every or last_step or step % save_every == 0 or (step >= warmup_steps and val_loss_accum.item() < best_val_loss):
                best_val_loss = min(best_val_loss, val_loss_accum.item())
                if best_val_loss < 5.56:  # 4.75 for Chirp, test (low data) Music was 5.225, proper Music 5.56?
                    val_loss_steps = 35
                    eval_every = 50
                elif best_val_loss < 5.73:  # 4.825 for Chirp, test (low data) Music was 5.365, proper Music 5.73?
                    val_loss_steps = 20
                    eval_every = 100
                else:
                    val_loss_steps = 10
                    eval_every = 200
                # Don't save on the first step when resuming
                if not resume or step != chkpt["step"]:
                    name = f"model_s{step:05d}_vl{val_loss_accum.item():.4f}.pt" if step % save_every == 0 else "model.pt"
                    checkpoint_path = os.path.join(log_dir, name)
                    print(f"writing checkpoint to {checkpoint_path}")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        "optim": optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'norms_window': norms_window,
                        'loss_window': loss_window,
                        "divergence_window": divergence_window,
                        "grok_start_step": grok_start_step
                    }
                    torch.save(checkpoint, checkpoint_path)
                    if step % save_every == 0:
                        try:
                            print("\nGenerating audio samples...")
                            torch.cuda.empty_cache()
                            audio_folder = os.path.join(log_dir, f"model_s{step:05d}_vl{val_loss_accum.item():.4f}")
                            os.makedirs(audio_folder, exist_ok=True)

                            tokenizer = AudioTokenizer(device=device)

                            # Get validation data for prefill
                            x_val, _ = val_loader.next_batch(False)
                            x_val = x_val[:inference_batch_size].to(device)

                            model.eval()
                            with torch.no_grad():
                                # 1. Third chunk prediction
                                try:
                                    print("\n\t3rd chunk...")
                                    prefill = x_val[:, :chunk_size * 2]
                                    chunk3_tokens = generate_tokens(model, chunk_size, inference_batch_size, prefill)
                                    #print(f"Tokens generated: {chunk3_tokens} (len {len(chunk3_tokens[0])}). Saving files.")
                                    print(f"Tokens generated: {len(chunk3_tokens[0])}. Saving files.")
                                    save_audio_files(chunk3_tokens, tokenizer, audio_folder, "chunk3")
                                    del chunk3_tokens
                                except Exception as e:
                                    print(f"\nError generating audio sample: {e}.")
                                    print(format_exc())

                                # 2. Second and third chunks prediction
                                try:
                                    print("\n\t2nd&3rd chunk...")
                                    prefill = x_val[:, :chunk_size]
                                    chunk23_tokens = generate_tokens(model, chunk_size * 2, inference_batch_size,
                                                                     prefill)
                                    #print(f"Tokens generated: {chunk23_tokens} (len {len(chunk23_tokens[0])}). Saving files.")
                                    print(f"Tokens generated: {len(chunk23_tokens[0])}. Saving files.")
                                    save_audio_files(chunk23_tokens, tokenizer, audio_folder, "chunk2and3")
                                    del chunk23_tokens
                                except Exception as e:
                                    print(f"\nError generating audio sample: {e}.")
                                    print(format_exc())

                                # 3. Full sequence
                                try:
                                    print("\n\tFull sequence...")
                                    full_tokens = generate_tokens(model, T, inference_batch_size)
                                    #print(f"Tokens generated: {full_tokens} (len {len(full_tokens[0])}). Saving files.")
                                    print(f"Tokens generated: {len(full_tokens[0])}. Saving files.")
                                    save_audio_files(full_tokens, tokenizer, audio_folder, "full")
                                    del full_tokens
                                except Exception as e:
                                    print(f"\nError generating audio sample: {e}.")
                                    print(format_exc())

                                print("\nDone.")
                            del tokenizer
                        except Exception as e:
                            print(f"\nError generating audio samples: {e}.")
                            print(format_exc())
                        finally:
                            torch.cuda.empty_cache()

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

        if grok_enabled and step >= warmup_steps + norms_window_size:
            if grok_start_step >= 0:
                warmup_factor = min(1.0, (step - grok_start_step) / grok_warmup_steps) ** 3
                alpha = grok_alpha * warmup_factor
                lamb = grok_lamb  # * warmup_factor
                grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)
                wandb.log({
                    "debug/grok_warmup_factor": warmup_factor,
                    "debug/grok_alpha": alpha,
                    "debug/grok_lamb": lamb,
                }, step=step)
            else:
                divergence = np.mean(divergence_window)
                if divergence > grok_start_divergence:
                    print(f"Starting FastGrok (mean divergence {divergence} > {grok_start_divergence}, the grok_start_divergence; will warm up grok for {grok_warmup_steps} steps).")
                    grok_start_step = step

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    clip_val = get_clip_value(norms_window, step)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
    norms_window.append(norm.item())
    loss_window.append(loss_accum.item())
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
            #print(f"Epoch {current_epoch}")
            t.set_description(f"Training epoch {current_epoch+1} of {num_epochs}", refresh=True)
        wandb.log({
            "etc/step": step,
            "etc/epoch": current_epoch,
            "etc/lr": lr,
            "etc/norm": norm.item(),
            "etc/clip_value": clip_val,
            "etc/toks_per_sec": tokens_per_sec,
            "train/loss": loss_accum.item(),
        }, step=step)

if ddp:
    destroy_process_group()
