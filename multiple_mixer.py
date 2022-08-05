import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.nn as jnn
import equinox as eqx
import einops

from equinox.nn import LayerNorm, MLP
from jax import vmap, jit, grad
from einops import rearrange, reduce, repeat

import matplotlib
import matplotlib.pyplot as plt
import timeit

def ps(obj):
    """
    Utility function for printing the shape of an object
    """
    print(jnp.shape(obj))

def cyclic_permutations(a):
    permutations = [a[idx:] + a[:idx] for idx in reversed(range(len(a)))]
    return permutations[-1:] + permutations[:-1]

jax.lax.reduce(jnp.array([1, 2, 3, 4]), 0, sum, 0)
jax.lax.reduce(jnp.array([1, 2, 3, 4]), jnp.array([0]), sum, 0)
jax.lax.scan(sum, 0, jnp.array([1, 2, 3, 4]))

def fori_loop(lower, upper, body_fun, init_val):
  val = lambda x: x
  for i in range(lower, upper):
    val = vmap(i, val)
  return val

def vmap_i(i, a):
    return vmap(a, i, i)
vmap_i(1, sum)(jnp.ones((3, 3)))
vmap(sum, 1, 1)(jnp.ones((3, 3)))

# y = y + jax.vmap(mixer0, 0, 0)(norm0(y))
# y = y + jax.vmap(mixer1, 1, 1)(norm1(y))
# y = y + jax.vmap(mixer2, 2, 2)(norm2(y))
# for i, mixer_i, norm_i in enumerate(zip(self.mixers, self.norms)):
#     y = y + jax.vmap(mixer_i, i, i)(norm_i(y))


# an N-Mixer is two for loops
# the inner for loop constructs the nested x = vmap(x, j, j), skipping the ith
# range_no_j = range(0, j) + range(j+1, N)
# itr = first(range_no_j)
# x = vmap(mixer_i, itr, itr)
# for j in 1:N without i:
#   itr = next(range_no_j)
#     x = vmap(x, j, j)
# the second

for i in (1 to N without j):
    vmap_without_i = vmap(mixers_i, i, i)
vmap(mixer1, 1, 1)(vmap(mixer0, 0, 0)(y))
# vmap(vmap, 0, 0)


def d(f, i):
    def inner(x):
        for _ in range(i):
            x = f(x)
        return x
    return x

class MixerBlock(eqx.Module):
    patch_mixer: eqx.nn.MLP
    hidden_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key
    ):
        tkey, ckey = jrd.split(key, 2)
        self.patch_mixer = eqx.nn.MLP(
            num_patches, num_patches, mix_patch_size, depth=1, key=tkey
        )
        self.hidden_mixer = eqx.nn.MLP(
            hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
        )
        self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
        self.norm2 = eqx.nn.LayerNorm((hidden_size, num_patches))

    def __call__(self, y):
        y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
        y = y + jax.vmap(self.hidden_mixer, 1, 1)(self.norm2(y))
        return y

# Or we could make the mixerblock a list of individual mixers and norms
class MixerBlockND(eqx.Module):
    mixers: list
    norms: list

    def __init__(self, dim_sizes, width_sizes, *, key):
        # assert len(dim_sizes) == len(width_sizes)
        keys = jrd.split(key, len(dim_sizes))
        self.mixers = [
            eqx.nn.MLP(dim_size, dim_size, width_size, depth=1, key=key)
            for dim_size, width_size, key in zip(dim_sizes, width_sizes, keys)
        ]
        self.norms = [eqx.nn.LayerNorm(dims) for dims in cyclic_permutations(dim_sizes)]
        # self.norms = [eqx.nn.LayerNorm(None, elementwise_affine=False) for _ in dim_sizes]

    def __call__(self, y):
        for mixer, norm in zip(self.mixers, self.norms):
            vmap
            y = vmap(vmap(mixer))(norm(y)) + y
        return y


# Call to mixer block:
for i, mixer, norm in enumerate(zip(self.mixers, self.norms)):
    def total_vmap(x):
        return vmap(x, i)
    y = vmap(vmap(mixer))(norm(y)) + y
# return y


# class MixerBlock(eqx.Module):
#     patch_mixer: eqx.nn.MLP
#     hidden_mixer: eqx.nn.MLP
#     norm1: eqx.nn.LayerNorm
#     norm2: eqx.nn.LayerNorm

#     def __init__(
#         self, num_patches, hidden_size, mix_patch_size, mix_hidden_size, *, key
#     ):
#         tkey, ckey = jrd.split(key, 2)
#         self.patch_mixer = eqx.nn.MLP(
#             num_patches, num_patches, mix_patch_size, depth=1, key=tkey
#         )
#         self.hidden_mixer = eqx.nn.MLP(
#             hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
#         )
#         self.norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
#         self.norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))

#     def __call__(self, y):
#         y = y + jax.vmap(self.patch_mixer)(self.norm1(y))
#         y = einops.rearrange(y, "c p -> p c")
#         y = y + jax.vmap(self.hidden_mixer)(self.norm2(y))
#         y = einops.rearrange(y, "p c -> c p")
#         return y

# # def main1():
# ### Main:
# key = jrd.PRNGKey(40)
# num_patches = 3
# mix_patch_size = 16
# hidden_size = 16
# mix_hidden_size = 32
# tkey, ckey = jrd.split(key, 2)
# patch_mixer = eqx.nn.MLP(
#         num_patches, num_patches, mix_patch_size, depth=1, key=tkey
#     )
# hidden_mixer = eqx.nn.MLP(
#         hidden_size, hidden_size, mix_hidden_size, depth=1, key=ckey
#     )
# norm1 = eqx.nn.LayerNorm((hidden_size, num_patches))
# norm2 = eqx.nn.LayerNorm((num_patches, hidden_size))
# norm2a = eqx.nn.LayerNorm((hidden_size, num_patches))

# y = jrd.uniform(key, (hidden_size, num_patches))
# print(f"Original: {jnp.shape(y)}")
# y1 = y + jax.vmap(patch_mixer)(norm1(y))
# print(f"After 1st Vmap: {jnp.shape(y)}")
# y2 = einops.rearrange(y1, "c p -> p c")
# print(f"after 1st rearrange: {jnp.shape(y2)}")
# y3 = y2 + jax.vmap(hidden_mixer)(norm2(y2))
# print(f"2nd Vmap: {jnp.shape(y3)}")
# y4 = einops.rearrange(y3, "p c -> c p")
# print(f"after 2nd rearrange: {jnp.shape(y4)}")
# y4

# # jnp.transpose(norm2a(y1)) == norm2(y2) # (true)
# ps(y1 + vmap(hidden_mixer, 1, 1)(norm2a(y1)))
# ps(tkn)
# Mix1 = MixerBlock(num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=key)
# MixNT = MixerBlock(num_patches, hidden_size, mix_patch_size, mix_hidden_size, key=key)

#     _, ckey = jrd.split(key, 2)
#     hidden_mixer = eqx.nn.MLP(
#             npatches, npatches, hidden_width2, depth=1, key=ckey
#         )
#     return vmap(HM)(y)
#     # return vmap(hidden_mixer)(y)
# main1()