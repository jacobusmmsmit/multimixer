from typing import List  # Python 3.7 compatability

import equinox as eqx
import jax.random as jr

from .utils import antivmap


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
    mixer_dimensions: List[int]
    blocks: List[MultiMixerBlock]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        mixer_dimensions,
        mixer_widths,
        num_blocks,
        *,
        dims_per_block=None,
        key,
    ):
        """**Arguments**
        - `mixer_dimensions`: The dimensions of the mixer during the backbone operation.
        - `mixer_widths`: The widths of each dimension's depth-1 MLP in the mixer .
        - `num_blocks`: The number of mixer blocks an input will go through.

        **Kwargs**
        - `dims_per_block`: Which dimensions' MLPs to apply per block, if None then all are applied (must have length
            == num_blocks)
        - `key`: A JAX PRNG key.
        """
        bkeys = jr.split(key, num_blocks)

        self.mixer_dimensions = mixer_dimensions
        self.blocks = [
            MultiMixerBlock(
                mixer_dimensions,
                mixer_widths,
                apply_dims,
                key=bkey,
            )
            for apply_dims, bkey in zip(dims_per_block, bkeys)
        ]
        self.norm = eqx.nn.LayerNorm(mixer_dimensions)

    def __call__(self, y):
        for block in self.blocks:
            y = block(y)
        return self.norm(y)
