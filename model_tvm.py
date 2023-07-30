import tvm

from tvm.relax.testing import nn
from tvm import relax, te, tir

from dataclasses import dataclass
import time

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
    Linear,
    Embedding,
    ModuleList,
)

@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias, eps=1e-5, dtype="float32"):
        super().__init__()
        self.dtype = dtype
        self.weight = nn.Parameter((ndim,), dtype=dtype, name="weight")
        self.bias = nn.Parameter((ndim,), dtype=dtype, name="bias") if bias else None
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

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, "float32", bias=config.bias)
        # output projection
        self.c_proj = Linear(config.n_embd, config.n_embd, "float32", bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
    def forward(self, x, past_kv=None, use_cache=False):
        # start_time = time.time()

        B, T, C = _size(x) # batch size, sequence length, embedding dimensionality (n_embd)

        def te_slice(x: te.Tensor, start: int, end: int):
            batch_size, seq_len, _ = x.shape
            return te.compute(
                shape=(batch_size, seq_len, end-start),
                fcompute=lambda i, j, k: x[i, j, start+k],
                name="slice"
            )
        
        query_key_value = self.c_attn(x)
        # query_key_value = x
        # print('Attention1: %.2f\n' % (time.time()-start_time))
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

        # if past_kv is not None:
        #     past_key = past_kv[0]
        #     past_value = past_kv[1]
        #     k = _cat((past_key, k), dim=-2)
        #     v = _cat((past_value, v), dim=-2)


        FULL_T = _size(k)[-2]

        # if use_cache is True:
        #     present = (k, v)
        # else:
        #     present = None
        present = None

        # if past_kv is not None:
        #     # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
        #     # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
        #     # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so 
        #     # to work around this we set is_causal=False.
        #     is_causal = False
        # else:
        #     is_causal = True
        is_causal = True

        # y = q
        y = _scaled_dot_product_attention(q, k, v, is_causal=is_causal)  
        y = _view(y, (B, T, C))
        y = self.c_proj(y)
        
        return (y, present)
    
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

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = nn.emit(x + attn_output)
        x = nn.emit(x + self.mlp(self.ln_2(x)))
        return (x, prev_kvs)

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = Embedding(config.input_vocab_size, config.n_embd, "float32")
        self.wpe = Embedding(config.block_size, config.n_embd, "float32")
        self.h = ModuleList([Block(config, idx) for idx in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = Linear(config.n_embd, config.output_vocab_size, "float32", bias=False)

    def forward(self, idx: relax.Expr, merge_len: relax.Expr, merge_context, past_kv, position_ids, use_cache):
        
        b, t = _size(idx)
        t_merge = t
        # t_merge = merge_len.struct_info.values[0] if merge_context else t

        if past_kv is not None:
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        else:
            # forward the GPT model itself
            if merge_context:
                t_merge = merge_len.struct_info.values[0]
                # t_merge = merge_len.value # changed

                def te_slice(x: te.Tensor, start: int, end: int):
                    batch_size, seq_len = x.shape
                    return te.compute(
                        shape=(batch_size, end - start),
                        fcompute=lambda i, j: x[i, start+j],
                        name="slice"
                    )
                def te_slice_1(x: te.Tensor, start: int):
                    batch_size, seq_len = x.shape
                    return te.compute(
                        shape=(batch_size, seq_len - start),
                        fcompute=lambda i, j: x[i, start+j],
                        name="slice_1"
                    )
                
                idx_1 = nn.emit_te(te_slice, idx, 0, 256, primfunc_name_hint="slice")
                idx_2 = nn.emit_te(te_slice, idx, 256, 256+256, primfunc_name_hint="slice")
                idx_3 = nn.emit_te(te_slice, idx, 256+256, t, primfunc_name_hint="slice")

                def extend_te(x: te.Tensor, y: te.Tensor):
                    a1, b1, c1 = x.shape
                    _, b2, _ = y.shape
                    return te.compute(
                        (a1, t_merge, c1),
                        lambda i1, i2, i3: te.if_then_else(
                            i2 < b1,
                            x[i1, i2, i3],
                            y[i1, i2-b1, i3],
                        ),
                        name="concat_te",
                    )

                tok_emb = nn.emit_te(extend_te, nn.emit(self.wte(idx_1) + self.wte(idx_2)), self.wte(idx_3), primfunc_name_hint="concat_te")
            else:
                tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)

        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.h))

        if position_ids is None:
            position_ids = _arange(past_length, t_merge + past_length, dtype="int64")
            position_ids = _unsqueeze(position_ids, 0) # shape (1, t)

        pos_emb = self.wpe(position_ids) # position embeddings of shape (1, t, n_embd)

        x = nn.emit(tok_emb + pos_emb)

        new_kv = () if use_cache else None

        for i, (block, past_layer_kv) in enumerate(zip(self.h, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            # if use_cache:
            #     new_kv = new_kv + (kv,)

        x = self.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        def te_slice_last(x: te.Tensor):
            _, seq_len, n_embd = x.shape
            return te.compute(
                shape=(1, 1, n_embd),
                fcompute=lambda i, _, k: x[i, seq_len - 1, k],
                name="slice_last",
            )

        
        x = nn.emit_te(
            te_slice_last,
            x,
            primfunc_name_hint="slice_last",
        )

        logits = self.lm_head(x) 

        return (logits, new_kv)
