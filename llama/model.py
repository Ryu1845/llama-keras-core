"""keras_core implementation of [Llama](https://arxiv.org/abs/2302.13971)
Based on [minimal-llama](https://github.com/zphang/minimal-llama/blob/main/minimal_llama/model.py)"""
import math

import keras_core as keras
from keras_core import ops, Layer

MULTIPLE_OF=256

class RMSNorm(Layer):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def build(self, input_shape):
        self.weight = self.add_weight(shape=input_shape, initializer="ones", trainable=True, name="weight")

    def call(self, inputs):
        def norm(x):
            return x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps)
        output = ops.cast(norm(ops.cast(inputs, dtype="float32")), dtype=inputs.dtype)
        return output * self.weight


class FeedForward(Layer):
    def __init__(self, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2*hidden_dim/3)
        self.hidden_dim = MULTIPLE_OF * ((hidden_dim+MULTIPLE_OF-1)//MULTIPLE_OF)

    def build(self, input_shape):
        self.w1 = keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.w2 = keras.layers.Dense(input_shape[-1], use_bias=False)
        self.w3 = keras.layers.Dense(self.hidden_dim, use_bias=False)

    def call(self, inputs):
        return self.w2(ops.silu(self.w1(inputs))*self.w3(inputs))


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return ops.concatenate((-x2, x1), axis=-1)
    return (q*cos) + (rotate_half(q) * sin), (k*cos) + (rotate_half(k) * sin)


class Attention(Layer):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def build(self, input_shape):
        head_dim = input_shape[-1]//self.n_heads
        self.head_dim = head_dim
        self.wq = keras.layers.Dense(self.n_heads*head_dim, use_bias=False)
        self.wk = keras.layers.Dense(self.n_heads*head_dim, use_bias=False)
        self.wv = keras.layers.Dense(self.n_heads*head_dim, use_bias=False)
        self.wo = keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, inputs, cos, sin, mask):
        bsz, seqlen, _ = inputs.shape
        xq, xk, xv = self.wq(inputs), self.wk(inputs), self.wv(inputs)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        keys = xk[:, :seqlen]
        values = xv[:, :seqlen]

        xq = ops.transpose(xq, axes=(1, 2))
        keys = ops.transpose(keys, axes=(1, 2))
        values = ops.transpose(values, axes=(1, 2))
        scores = ops.matmul(xq, ops.transpose(keys, axes=(2,3)))/math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask # (bsz, n_heads, seqlen, cache_len + seqlen)
        scores = ops.cast(ops.softmax(ops.cast(scores, dtype="float32"), axis=-1), dtype=xq.dtype)
        output = ops.matmul(scores, values)  # (bsz, n_heads, seqlen, head_dim)
        output = ops.transpose(output, axes=(1, 2)).view(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlock(Layer):
    def __init__(self, n_heads: int, norm_eps: float = 1e-6):
        super().__init__()
        self.n_heads=n_heads
        self.norm_eps=norm_eps

    def build(self, input_shape):
        self.attention = Attention(n_heads=self.n_heads)
        self.feed_forward = FeedForward(hidden_dim=4*input_shape[-1])
        self.attention_norm = RMSNorm(eps=self.norm_eps)
        self.ffn_norm = RMSNorm(eps=self.norm_eps)

    def call(self, inputs, cos, sin, mask):
        h = inputs + self.attention(self.attention_norm(inputs), cos, sin, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

def get_model():
    pass


if __name__ == '__main__':
    model = get_model()
  
