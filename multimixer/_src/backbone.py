from typing import List, Optional, Sequence  # Python 3.7 compatability

import equinox as eqx
import jax.random as jr

from .utils import antivmap


# from jaxtyping import Array, Float


class MultiMixerBlock(eqx.Module):
    """Maps a different MLP over dims of the input specified by apply_dims."""

    mixers: List[eqx.nn.MLP]
    norms: List[eqx.nn.LayerNorm]
    apply_dims: List[int]

    def __init__(
        self,
        mixer_dimensions: Sequence[int],
        mlp_widths: Sequence[int],
        *,
        apply_dims: Optional[Sequence[Sequence[int]]] = None,
        key,
    ):
        """**Arguments:**
        - `mixer_dimensions`: The mixer_dimensions of the input and output.
        - `mlp_widths`: The width of the single hidden layer in the MLP of each dimension.

        **Kwargs**
        - `apply_dims`: Which dimensions to apply an MLP down, defaults to all
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        # mixer_dimensions are put in from local to global
        assert len(mixer_dimensions) == len(mlp_widths)
        mlp_keys = jr.split(key, len(mixer_dimensions))

        mixers = []
        norms = []
        for input_dim, (mixer_dim, mlp_width, mlp_key) in enumerate(
            zip(mixer_dimensions, mlp_widths, mlp_keys)
        ):
            if input_dim in apply_dims:
                mixers.append(
                    eqx.nn.MLP(mixer_dim, mixer_dim, mlp_width, depth=1, key=mlp_key)
                )
                norms.append(eqx.nn.LayerNorm(mixer_dimensions))

        self.mixers = mixers
        self.norms = norms
        self.apply_dims = apply_dims  # Example: [0, 1]

    def __call__(self, y):  # : Float[Array, " *self.mixer_dimensions"]
        """**Arguments**
        - `y`: The input. Should be of shape `(mixer_dimensions)`.
        """
        for i, mixer, norm in zip(self.apply_dims, self.mixers, self.norms):
            y = y + antivmap(mixer, i)(norm(y))
        return y


class MultiMixer(eqx.Module):
    """An MLP-Mixer with multiple patch dimensions/scales"""

    blocks: List[MultiMixerBlock]
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        mixer_dimensions: Sequence[
            int
        ],  # Should we keep the input_size for assertions?
        mixer_widths: Sequence[int],
        num_blocks: Sequence[int],
        *,
        dims_per_block: Optional[Sequence[Sequence[int]]] = None,
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
        if dims_per_block is None:
            dims_per_block = [
                list(range(len(mixer_dimensions))) for _ in range(num_blocks)
            ]

        self.blocks = [
            MultiMixerBlock(
                mixer_dimensions,
                mixer_widths,
                apply_dims=apply_dims,
                key=bkey,
            )
            for apply_dims, bkey in zip(dims_per_block, bkeys)
        ]
        self.norm = eqx.nn.LayerNorm(mixer_dimensions)

    def __call__(self, y):  # : Float[Array, " *self.mixer_dimensions"]
        for block in self.blocks:
            y = block(y)
        return self.norm(y)
