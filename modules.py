from tvm.relax.testing import nn
from tvm import relax, tir

def shape_of(tensor):
    if isinstance(tensor, relax.Expr):
        if not isinstance(tensor.struct_info, relax.TensorStructInfo):
            raise TypeError("The input Expr of shape_of should be a Tensor")
        return tensor.struct_info.shape
    raise ValueError("Unsupported type: {}".format(type(tensor)))

def _layer_norm(*args, **kwargs) -> relax.Var:
    x = args[0]
    
    normalized_shape = args[1]
    
    dim_num = len(normalized_shape)
    axes = list(range(-dim_num, 0))
    
    gamma = kwargs["weight"]
    beta = kwargs["bias"]
    
    dtype = x.struct_info.dtype
    
    if beta is None:
        # shape_tuple = [int(s) for s in normalized_shape.values]
        beta = relax.op.full(normalized_shape, relax.const(0, dtype), dtype)
        # beta = relax.const(np.zeros(normalized_shape), x.struct_info.dtype)
    
    eps = kwargs["eps"]
    
    return nn.emit(relax.op.nn.layer_norm(
        x,
        gamma,
        beta,
        axes=axes,
        epsilon=eps,
    ))

def _gelu(*args, **kwargs) -> relax.Var:
    if "approximate" not in kwargs:
            approximate = "none"
    else:
        approximate = kwargs["approximate"]
    if approximate == "none":
        return nn.emit(relax.op.nn.gelu(args[0]))
    elif approximate == "tanh":
        return nn.emit(relax.op.nn.gelu_tanh(args[0]))
    else:
        raise KeyError("Unregonized approximate algorithm for gelu: {}.".format(approximate))

def _size(*args, **kwargs) -> relax.Expr:
    x = args[0]
    shape = shape_of(x)
    if len(args) == 1:
        assert isinstance(shape, relax.ShapeExpr)
        return shape
    assert len(args) == 2
    idx = args[1]
    return shape[idx]

def _split(*args, **kwargs) -> relax.Var:
    x = args[0]
    split_size = args[1]
    if "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        dim = 0
    n_section = (shape_of(x)[dim].value + split_size - 1) // split_size

    return nn.emit(relax.op.split(x, n_section, dim))

def _reshape(*args, **kwargs) -> relax.Expr:
    return relax.op.reshape(args[0], tuple(args[1]))

def _view(*args, **kwargs) -> relax.Var:
    return nn.emit(_reshape(*args, **kwargs))

def _transpose(*args, **kwargs) -> relax.Var:
    full_idx = list(range(len(shape_of(args[0]))))
    full_idx[args[1]], full_idx[args[2]] = full_idx[args[2]], full_idx[args[1]]
    return nn.emit(relax.op.permute_dims(args[0], full_idx))

def _cat(*args, **kwargs) -> relax.Var:
    return nn.emit(relax.op.concat(args[0], axis=kwargs["dim"]))

def _scaled_dot_product_attention(*args, **kwargs) -> relax.Var:
    assert (
        len(args) <= 4
    ), "Dropout is not supported, and is_causal should be called by kwargs."
    transpose_S_H = lambda tensor: relax.op.permute_dims(tensor, [0, 2, 1, 3])
    query = transpose_S_H(args[0])
    key = transpose_S_H(args[1])
    value = transpose_S_H(args[2])
    causal_mask = "TopLeft" if kwargs.get("is_causal", False) else None

    if len(args) == 4:
        mask = args[3]
        msg = "Only a float mask is supported for the attn_mask input."
        assert "float" in mask.struct_info.dtype, msg
        attn = relax.op.nn.attention(query, key, value, bias=mask, causal_mask=causal_mask)
    else:
        attn = relax.op.nn.attention(query, key, value, causal_mask=causal_mask)

    return nn.emit(attn)

def _contiguous(*args, **kwargs) -> relax.Var:
    return args[0]

def _arange(*args, **kwargs) -> relax.Var:
    start_end_step = [None, None, None]
    if "start" in kwargs:
        start_end_step[0] = kwargs["start"]
    if "end" in kwargs:
        start_end_step[1] = kwargs["end"]
    if "step" in kwargs:
        start_end_step[2] = kwargs["step"]

    if len(args) == 1:
        assert start_end_step[1] is None
        start_end_step[1] = args[0]
    elif len(args) == 2:
        assert start_end_step[0] is None
        assert start_end_step[1] is None
        start_end_step[0] = args[0]
        start_end_step[1] = args[1]
    elif len(args) == 3:
        assert start_end_step[0] is None
        assert start_end_step[1] is None
        assert start_end_step[2] is None
        start_end_step[0] = args[0]
        start_end_step[1] = args[1]
        start_end_step[2] = args[2]

    if start_end_step[0] is None:
        start_end_step[0] = 0
    if start_end_step[2] is None:
        start_end_step[2] = 1

    dtype = "int64"

    return relax.op.arange(*start_end_step, dtype=dtype)

def _unsqueeze(*args, **kwargs) -> relax.Var:
    return nn.emit(
        relax.op.expand_dims(args[0], args[1])
    )

def _sum(*args, **kwargs) -> relax.Var:
    x = args[0]
    axis = kwargs["dim"] if "dim" in kwargs else None
    keepdims = kwargs["keepdim"] if "keepdim" in kwargs else False

    return nn.emit(relax.op.sum(x, axis, keepdims))


from tvm import relax, topi, tir

class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dtype,
        bias=True,
        out_dtype=None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            (out_features, in_features),
            dtype=dtype,
            name="linear_weight",
        )
        if bias:
            self.bias = nn.Parameter(
                (out_features,),
                dtype=dtype if out_dtype is None else out_dtype,
                name="linear_bias",
            )
        else:
            self.bias = None
        self.dtype = dtype
        self.out_dtype = out_dtype

    def forward(self, x: relax.Expr) -> relax.Var:
        x = nn.emit(relax.op.linear(x, self.weight, self.bias, out_dtype="float32"))
        
        # x = nn.emit(x)
        # weight = relax.op.permute_dims(self.weight, axes=None)
        # x = nn.emit(relax.op.matmul(x, weight, out_dtype=self.out_dtype))
        # if self.bias is not None:
        #     x = nn.emit(x + self.bias)

        # x = nn.emit_te(topi.matmul, x, weight)
        # if self.bias is not None:
        #     x = nn.emit_te(topi.add, x, self.bias)

        return x

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype=dtype, name="weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(relax.op.take(self.weight, x, axis=0))
        x_shape = x.struct_info.shape.values
        emb_size = self.weight.struct_info.shape.values[-1]
        x = nn.emit(relax.op.reshape(x, shape=[-1]))
        embedding = nn.emit(relax.op.take(self.weight, x, axis=0))
        return nn.emit(relax.op.reshape(embedding, [*x_shape, emb_size]))

class ModuleList(nn.Module):
    def __init__(self, modules):
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def forward(self, x: relax.Expr) -> relax.Var:
        for module in self.modules:
            x = module(x)
        return x
