import tvm

from tvm.relax.testing import nn
from tvm import relax, te, tir

from dataclasses import dataclass

import modules
from modules import (
    _layer_norm,
    _gelu,
    _size,
    _split,
    _view,
    _transpose,
    _cat,
    _scaled_dot_product_attention,
    _contiguous,
    _arange,
    _unsqueeze,
    _sum,
    Linear,
    Embedding,
    ModuleList,
)

@dataclass
class FineGPTConfig:
    n_codes_total: int = 8
    n_codes_given: int = 1
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False, eps=1e-5, dtype="float32"):
        super().__init__()
        self.dtype = dtype
        self.weight = nn.Parameter((ndim,), dtype=dtype, name="weight_ln")
        self.bias = nn.Parameter((ndim,), dtype=dtype, name="bias_ln") # if bias else None
        self.eps = eps

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(relax.op.astype(x, self.dtype))
        x = _layer_norm(
                x,
                self.weight.struct_info.shape,
                weight = self.weight,
                bias = self.bias,
                eps = 1e-5
            )
        return x
    
class GELU(nn.Module):
    
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate
        
    def forward(self, input):
        return _gelu(input, approximate=self.approximate)
    
class NonCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, "float32", bias=config.bias)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd, "float32", bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        
        B, T, C = _size(x)  # batch size, sequence length, embedding dimensionality (n_embd)

        def te_slice(x: te.Tensor, start: int, end: int):
            batch_size, seq_len, _ = x.shape
            return te.compute(
                shape=(batch_size, seq_len, end-start),
                fcompute=lambda i, j, k: x[i, j, start+k],
                name="slice"
            )
        
        query_key_value = self.c_attn(x)
        
        q = nn.emit_te(
            te_slice,
            query_key_value,
            0,
            self.n_embd,
            primfunc_name_hint="slice"
        )
        k = nn.emit_te(
            te_slice,
            query_key_value,
            self.n_embd,
            self.n_embd*2,
            primfunc_name_hint="slice"
        )
        v = nn.emit_te(
            te_slice,
            query_key_value,
            self.n_embd*2,
            self.n_embd*3,
            primfunc_name_hint="slice"
        )
        
        k = _transpose(_view(k, (B, T, self.n_head, C // self.n_head)), 1, 2) # (B, nh, T, hs)
        q = _transpose(_view(q, (B, T, self.n_head, C // self.n_head)), 1, 2) # (B, nh, T, hs)
        v = _transpose(_view(v, (B, T, self.n_head, C // self.n_head)), 1, 2) # (B, nh, T, hs)

        y = _scaled_dot_product_attention(q, k, v, is_causal=False) 
        y = _view(y, (B, T, C))
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = Linear(config.n_embd, 4 * config.n_embd, "float32", bias=config.bias)
        self.c_proj  = Linear(4 * config.n_embd, config.n_embd, "float32", bias=config.bias)
        self.gelu = GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class FineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = nn.emit(x + self.attn(self.ln_1(x)))
        x = nn.emit(x + self.mlp(self.ln_2(x)))
        return x
    
class FineGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.n_codes_total = config.n_codes_total
        
        self.wtes=ModuleList(
            [
                Embedding(config.input_vocab_size, config.n_embd, "float32")
                for _ in range(config.n_codes_total)
            ]
        )
        self.wpe=Embedding(config.block_size, config.n_embd, "float32")
        self.h=ModuleList([FineBlock(config) for _ in range(config.n_layer)])
        self.ln_f=LayerNorm(config.n_embd)

        self.lm_heads = ModuleList(
            [
                Linear(config.n_embd, config.output_vocab_size, "float32", bias=False)
                for _ in range(config.n_codes_given, self.n_codes_total)
            ]
        )
        # for i in range(self.n_codes_total - config.n_codes_given):
        #     # print(dir(self.lm_heads[i].weight))
        #     # print(self.lm_heads[i].weight.struct_info)
        #     self.wtes[i + 1].weight = relax.Var("wte_finegpt", self.lm_heads[i].weight.struct_info)

    def forward(self, pred_idx_shape, idx):

        b, t, codes = _size(idx)
        pred_idx = pred_idx_shape.struct_info.values[0]

        pos = _arange(0, t, dtype="int64")
        pos = _unsqueeze(pos, 0)

        # # forward the GPT model itself
        def te_slice(x: te.Tensor, slice_idx: int):
            dim0, dim1, _ = x.shape
            return te.compute(
                shape=(dim0, dim1),
                fcompute=lambda i, j: x[i, j, slice_idx],
                name="slice"
            )
        
        tok_embs = [
            _unsqueeze(wte(nn.emit_te(te_slice, idx, i, primfunc_name_hint="slice")), -1)
            for i, wte in enumerate(self.wtes)
        ]

        tok_emb = _cat(tok_embs, dim=-1)
        
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        
        def te_slice_1(x: te.Tensor, end: int):
            dim0, dim1, dim2, _ = x.shape
            return te.compute(
                shape=(dim0, dim1, dim2, end),
                fcompute=lambda i, j, k, l: x[i, j, k, l],
                name="slice_1"
            )
        
        print(_size(nn.emit_te(te_slice_1, tok_emb, pred_idx + 1, primfunc_name_hint="slice_1")))

        x = _sum(nn.emit_te(te_slice_1, tok_emb, pred_idx + 1, primfunc_name_hint="slice_1"), dim=-1)

        x = nn.emit(x + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        
        logits = [lm(x) for lm in self.lm_heads]
        return logits
