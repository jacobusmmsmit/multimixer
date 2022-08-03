import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.nn as jnn
import equinox as eqx
from equinox.nn import LayerNorm, MLP

from jax import vmap, jit, grad
from einops import rearrange, reduce, repeat

### MLP-Mixer:
# Visually
# input is of size (M, N) Channels by Patches
# input -> eqx.nn.LayerNorm -> (M, N) => (N, M) -> MLP1 -> (N, M) => (M, N) ->+ -> O -> LayerNorm -> MLP2 -> + -> output
#  V                                                                          ^    V                         ^
# skip-connection ----------------------------->------------------------------^    skip-connection ----->----^

# In Factory, Module, Op notation
# Setup: M, N, DS, DC = 16, 3, 256, 2048
# MLP1 ~ MLP([M, DS, DS, M], gelu)
# MLP2 ~ MLP([N, DC, DC, N], gelu)
# LN ~ LayerNorm()
# Tr ~ Transpose()
# Leading to the two components
# f: x -> (TR.vmap(MLP1).TR.LN)(x) + x
# g: x -> (vmap(MLP2).LN)(x) + x
# MLPMixer(x) = (g.f)(x)


def accumulate(f, x):
    """
    A skip connection
    """
    return f(x) + x


def ps(obj):
    """
    Utility function for printing the shape of an object
    """
    print(jnp.shape(obj))


class MLPMixer(eqx.Module):
    MLP1: eqx.Module
    MLP2: eqx.Module
    LN: eqx.Module

    def __init__(self, nchannels, npatches, hidden_width1, hidden_width2, key):
        mlp1_key, mlp2_key = jrd.split(key)
        self.MLP1 = MLP(nchannels, nchannels, hidden_width1, 2, jnn.gelu, key=mlp1_key)
        self.MLP2 = MLP(npatches, npatches, hidden_width2, 2, jnn.gelu, key=mlp2_key)
        self.LN = LayerNorm(None, elementwise_affine=False)

    def __call__(self, x):
        def mix(x):
            """
            Mix = Tr ∘ MLP1 ∘ Tr ∘ LN
            """
            z0 = self.LN(x)
            z1 = jnp.transpose(z0)
            z2 = vmap(self.MLP1)(z1)
            z3 = jnp.transpose(z2)
            return z3

        def nomix(x):
            z0 = self.LN(x)
            z1 = vmap(self.MLP2)(z0)
            return z1

        z1 = accumulate(mix, x)
        y = accumulate(nomix, z1)
        return y


if __name__ == "__main__":
    key = jrd.PRNGKey(42)
    mlp1_key, mlp2_key = jrd.split(key)
    nchannels = 4
    npatches = 3
    hidden_width1 = 16
    hidden_width2 = 32

    Mix1 = MLPMixer(nchannels, npatches, hidden_width1, hidden_width2, key)
    tkn = jrd.uniform(key, (nchannels, npatches))
    print(Mix1(tkn))
