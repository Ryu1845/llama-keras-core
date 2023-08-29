"""keras_core implementation of [Llama](https://arxiv.org/abs/2302.13971)
Based on [minimal-llama](https://github.com/zphang/minimal-llama/blob/main/minimal_llama/model.py)"""
import keras_core as keras
from keras_core import ops, Layer


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
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.w1 = keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.w2 = keras.layers.Dense(self.dim, use_bias=False)
        self.w3 = keras.layers.Dense(self.hidden_dim, use_bias=False)

    def call(self, inputs):
        return self.w2(ops.silu(self.w1(inputs))*self.w3(inputs))


def get_model():
    pass


if __name__ == '__main__':
    model = get_model()
  
