import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from gpt2 import GPTConfig


class Block(nn.Module):
    def __init__(self, config, input_width_multiplier=1.0, output_width_multiplier=1.0):
        super().__init__()
        self.input_n_embd = int(config.n_embd * input_width_multiplier)
        self.output_n_embd = int(config.n_embd * output_width_multiplier)

        self.ln_1 = nn.LayerNorm(self.input_n_embd)
        self.attn = CausalSelfAttention(config, input_width_multiplier, output_width_multiplier)
        self.ln_2 = nn.LayerNorm(self.output_n_embd)
        self.mlp = MLP(config, output_width_multiplier)

        # Add a dimension reduction layer if input and output widths differ
        if self.input_n_embd != self.output_n_embd:
            self.dim_reduction = nn.Linear(self.input_n_embd, self.output_n_embd)
        else:
            self.dim_reduction = nn.Identity()

    def forward(self, x):
        # Self-attention
        attn_output = self.attn(self.ln_1(x))
        x = self.dim_reduction(x) + attn_output

        # MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config, input_width_multiplier=1.0, output_width_multiplier=1.0):
        super().__init__()
        self.input_n_embd = int(config.n_embd * input_width_multiplier)
        self.output_n_embd = int(config.n_embd * output_width_multiplier)
        assert self.input_n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.input_n_embd, 3 * self.input_n_embd)
        # output projection
        self.c_proj = nn.Linear(self.input_n_embd, self.output_n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.input_n_embd, dim=2)
        head_size = self.input_n_embd // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, self.input_n_embd)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, width_multiplier=1.0):
        super().__init__()
        self.n_embd = int(config.n_embd * width_multiplier)
        self.c_fc    = nn.Linear(self.n_embd, 4 * self.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, init_weights=False):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd * 2),
            wpe=nn.Embedding(config.block_size, config.n_embd * 2),
            h=nn.ModuleList([self._create_block(config, i) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        if init_weights:
            self.apply(self._init_weights)

    def _create_block(self, config, layer_index):
        if layer_index == 0:
            return Block(config, input_width_multiplier=2.0, output_width_multiplier=2.0)
        elif layer_index == 1:
            return Block(config, input_width_multiplier=2.0, output_width_multiplier=1.5)
        elif layer_index == 2:
            return Block(config, input_width_multiplier=1.5, output_width_multiplier=1.0)
        else:
            return Block(config, input_width_multiplier=1.0, output_width_multiplier=1.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, log=False):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if log:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if log:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer