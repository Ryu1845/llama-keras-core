"""keras_core implementation of [Llama](https://arxiv.org/abs/2302.13971)
Based on [minimal-llama](https://github.com/zphang/minimal-llama/blob/main/minimal_llama/model.py)"""
import math
import os
from typing import Callable, TypedDict

os.environ["KERAS_BACKEND"] = "jax"

import keras_core as keras
from keras_core import ops, Layer, Model

MULTIPLE_OF = 256


class BaseLayerKwargs(TypedDict):
    activity_regularizer: Callable
    trainable: bool
    dtype: str
    autocast: bool
    name: str


class RMSNorm(Layer):
    def __init__(self, eps: float = 1e-6, **layer_kwargs: BaseLayerKwargs):
        super().__init__(**layer_kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=input_shape, initializer="ones", trainable=True, name="weight"
        )

    def call(self, inputs):
        def norm(x):
            return x * ops.rsqrt(
                ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps
            )

        output = ops.cast(norm(ops.cast(inputs, dtype="float32")), dtype=inputs.dtype)
        return output * self.weight


class FeedForward(Layer):
    def __init__(self, hidden_dim: int, **layer_kwargs: BaseLayerKwargs):
        super().__init__(**layer_kwargs)
        hidden_dim = int(2 * hidden_dim / 3)
        self.hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) // MULTIPLE_OF)

    def build(self, input_shape):
        self.w1 = keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.w2 = keras.layers.Dense(input_shape[-1], use_bias=False)
        self.w3 = keras.layers.Dense(self.hidden_dim, use_bias=False)

    def call(self, inputs):
        return self.w2(ops.silu(self.w1(inputs)) * self.w3(inputs))


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return ops.concatenate((-x2, x1), axis=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(Layer):
    def __init__(self, n_heads: int, **layer_kwargs: BaseLayerKwargs):
        super().__init__(**layer_kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        head_dim = input_shape[-1] // self.n_heads
        self.head_dim = head_dim
        self.wq = keras.layers.Dense(self.n_heads * head_dim, use_bias=False)
        self.wk = keras.layers.Dense(self.n_heads * head_dim, use_bias=False)
        self.wv = keras.layers.Dense(self.n_heads * head_dim, use_bias=False)
        self.wo = keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, inputs, cos, sin, mask):
        bsz, seqlen, _ = inputs.shape
        xq, xk, xv = self.wq(inputs), self.wk(inputs), self.wv(inputs)
        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        keys = xk[:, :seqlen]
        values = xv[:, :seqlen]

        xq = ops.transpose(xq, axes=(0, 2, 1, 3))
        keys = ops.transpose(keys, axes=(0, 2, 1, 3))
        values = ops.transpose(values, axes=(0, 2, 1, 3))
        scores = ops.matmul(xq, ops.transpose(keys, axes=(0, 1, 3, 2))) / math.sqrt(
            self.head_dim
        )
        if mask is not None:
            scores = scores + mask  # (bsz, n_heads, seqlen, cache_len + seqlen)
        scores = ops.cast(
            ops.softmax(ops.cast(scores, dtype="float32"), axis=-1), dtype=xq.dtype
        )
        output = ops.matmul(scores, values)  # (bsz, n_heads, seqlen, head_dim)
        output = ops.transpose(output, axes=(0, 2, 1, 3)).reshape(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlock(Layer):
    def __init__(
        self, n_heads: int, norm_eps: float = 1e-6, **layer_kwargs: BaseLayerKwargs
    ):
        super().__init__(**layer_kwargs)
        self.n_heads = n_heads
        self.norm_eps = norm_eps

    def build(self, input_shape):
        self.attention = Attention(n_heads=self.n_heads)
        self.feed_forward = FeedForward(hidden_dim=4 * input_shape[-1])
        self.attention_norm = RMSNorm(eps=self.norm_eps)
        self.ffn_norm = RMSNorm(eps=self.norm_eps)

    def call(self, inputs, cos, sin, mask):
        h = inputs + self.attention(self.attention_norm(inputs), cos, sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


def precompute_cos_sin(seq_len: int, dim: int, dtype, base: int = 10_000):
    inv_freq = 1.0 / (
        base ** (ops.cast(ops.arange(start=0, stop=dim, step=2), dtype="float32") / dim)
    )
    t = ops.arange(seq_len, dtype=dtype)
    freqs = ops.einsum("i,j->ij", t, inv_freq)
    emb = ops.concatenate((freqs, freqs), axis=-1)
    cos = ops.cos(emb)[None, :, None, :]
    sin = ops.sin(emb)[None, :, None, :]
    return cos, sin


class Transformer(Model):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_layers: int,
        dim: int,
        n_heads: int,
        norm_eps: float,
        **layer_kwargs: BaseLayerKwargs,
    ):
        super().__init__(**layer_kwargs)
        self.tok_embeddings = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=dim, name="token_embeddings"
        )
        self.blocks = [
            TransformerBlock(n_heads=n_heads, norm_eps=norm_eps, **layer_kwargs)
            for _ in range(n_layers)
        ]
        self.norm = RMSNorm(eps=norm_eps)
        self.lm_head = keras.layers.Dense(vocab_size, use_bias=False, name="lm_head")
        self.cos_cached, self.sin_cached = precompute_cos_sin(
            max_seq_len, dim // n_heads, dtype="float32"
        )

    def call(self, inputs, training=False):
        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)
        cos = ops.cast(self.cos_cached[:, :seqlen], dtype=h.dtype)
        sin = ops.cast(self.sin_cached[:, :seqlen], dtype=h.dtype)

        mask = ops.full((1, 1, seqlen, seqlen), fill_value=-1e10)
        mask = ops.triu(mask, k=1)

        for block in self.blocks:
            h = block(h, cos, sin, mask)
        h = self.norm(h)
        output = self.lm_head(h)
        return output


if __name__ == "__main__":
    model = Transformer(
        vocab_size=2**10,
        max_seq_len=128,
        n_layers=4,
        dim=384,
        n_heads=8,
        norm_eps=1e-7,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseTopKCategoricalAccuracy()],
    )

    class MyDataset(keras.utils.PyDataset):
        def __len__(self):
            return 2**14

        def __getitem__(self, idx):
            return ops.array([[idx % 1024]]), ops.array([[(idx + 1) % 1024]])

    model.fit(MyDataset(), epochs=2, batch_size=256)
    model.summary()
