from typing import List  # Python 3.7 compatability

import equinox as eqx
import jax.random as jr

from .utils import (
    antivmap,
    get_npatches,
    multi_patch_rearrange,
    reverse_multi_patch_rearrange,
    verify_patches,
)


class MultiMixerBlock(eqx.Module):
    """Maps a different MLP over each dimension of the input from first to last."""

    mixers: List[eqx.nn.MLP]
    norms: List[eqx.nn.LayerNorm]

    def __init__(self, dimensions, mlp_widths, *, key):
        """**Arguments:**
        - `dimensions`: The dimensions of the input and output.
        - `mlp_widths`: The width of the single hidden layer in the MLP of each dimension.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        # dimensions are put in from local to global
        assert len(dimensions) == len(mlp_widths)
        mlp_keys = jr.split(key, len(dimensions))
        self.mixers = [
            eqx.nn.MLP(dim, dim, mlp_width, depth=1, key=mlp_key)
            for dim, mlp_width, mlp_key in zip(dimensions, mlp_widths, mlp_keys)
        ]
        self.norms = [eqx.nn.LayerNorm(dimensions) for _ in dimensions]

    def __call__(self, y):
        """**Arguments**
        - `y`: The input. Should be of shape `(dimensions)`.
        """
        # TODO: improve compilation time by structured control flow primitives
        # something like: lax.fori_loop(0, len(self.mixers), vmap(mixer[i], i, i)(norm[i]), y)
        for i, (mixer, norm) in enumerate(zip(self.mixers, self.norms)):
            y = y + antivmap(mixer, i)(norm(y))
        return y


class MultiMixer(eqx.Module):
    """An MLP-Mixer with multiple patch dimensions/scales"""

    input_shape: List[int]
    n_patches: List[int]
    patch_sizes: List[int]
    hidden_size: int
    projection_in: eqx.nn.Linear
    projection_out: eqx.nn.Linear
    blocks: List[MultiMixerBlock]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        input_shape,
        patch_sizes,
        hidden_size,
        mix_patch_sizes,
        mix_hidden_size,
        num_blocks,
        *,
        out_channels=None,
        key,
    ):
        """**Arguments**
        - `input_shape`: The size of the input.
        - `n_patches`: The number of patches contained inside a single patch of the previous dimension (or the whole
            image for the first).
        - `patch_sizes`: The side length of the square patches for each patch scale from largest to smallest.
        - `hidden_size`: The number of channels each pixel will have during the mixing (?). A higher number potentially
            means more information can be transferred.
        - `mix_patch_sizes`: The number of hidden layers in the MLP corresponding to each patch scale.
        - `mix_hidden_size`: The number of hidden layers in the MLP corresponding to the "hidden" channels.
        - `num_blocks`: The number of mixer blocks an input will go through.

        **Kwargs**
        - `out_channels`: The number of channels in the output.
        - `key`: A JAX PRNG key.
        """
        channels, height, width = input_shape
        if out_channels is None:
            out_channels = channels
        assert height == width  # Currently square patches only, rectangular planned
        n_patches = get_npatches(width, patch_sizes)  # largest to smallest

        # for patch[i], n*size == size of patch[i-1]
        assert verify_patches(width, patch_sizes, n_patches)
        n_patches_square = [n**2 for n in n_patches]
        mixer_dimensions = tuple(reversed((*n_patches_square, hidden_size)))
        mixer_widths = tuple(reversed((*mix_patch_sizes, mix_hidden_size)))

        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.input_shape = input_shape
        self.n_patches = n_patches
        self.patch_sizes = patch_sizes
        self.hidden_size = hidden_size
        self.projection_in = eqx.nn.Linear(
            channels * patch_sizes[-1] ** 2, hidden_size, key=inkey
        )
        self.projection_out = eqx.nn.Linear(
            hidden_size, out_channels * patch_sizes[-1] ** 2, key=outkey
        )
        self.blocks = [
            MultiMixerBlock(
                mixer_dimensions,
                mixer_widths,
                key=bkey,
            )
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm(mixer_dimensions)

    def __call__(self, y):
        assert y.shape == self.input_shape
        y = multi_patch_rearrange(y, self.n_patches, self.patch_sizes)
        y = antivmap(self.projection_in)(y)  # nested jax.vmap
        for block in self.blocks:
            y = block(y)
        y = self.norm(y)
        y = antivmap(self.projection_out)(y)
        return reverse_multi_patch_rearrange(y, self.n_patches, self.patch_sizes)
