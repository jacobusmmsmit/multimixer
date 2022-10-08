from typing import Callable, Optional, Sequence

import equinox as eqx
import jax.random as jr

from .backbone import MultiMixer
from .utils import (
    antivmap,
    get_npatches,
    multi_patch_rearrange,
    reverse_multi_patch_rearrange,
    verify_patches,
)


class ImageMixer(eqx.Module):
    """A MultiMixer for 3D tensors"""

    mixer: MultiMixer
    rearrange_in: Callable
    rearrange_out: Callable
    projection_in: eqx.nn.Linear
    projection_out: eqx.nn.Linear

    def __init__(
        self,
        image_size: Sequence[int],
        patch_sizes: Sequence[int],
        hidden_size: int,
        patch_mlp_widths: Sequence[int],
        hidden_mlp_width: int,
        num_blocks: int,
        *,
        out_channels: Optional[int] = None,
        dims_per_block: Optional[Sequence[Sequence[int]]] = None,
        key,
    ):
        """**Arguments**
        - `n_patches`: The number of patches contained inside a single patch of the previous dimension (or the whole
            image for the first).
        - `patch_sizes`: The side length of the square patches for each patch scale from largest to smallest.
        - `hidden_size`: The number of channels each pixel will have during the mixing (?). A higher number potentially
            means more information can be transferred.
        - `patch_mlp_widths`: The widths of each patch's depth-1 MLP in the mixer.
        - `hidden_mlp_width`: The width of the hidden dimension's depth-1 MLP in the mixer.
        - `num_blocks`: The number of mixer blocks an input will go through.

        **Kwargs**
        - `out_channels`: The number of channels in the output.
        - `dims_per_block`: Which dimensions' MLPs to apply per block, if None then all are applied (must have length
            == num_blocks)
        - `key`: A JAX PRNG key.
        """

        channels, height, width = image_size
        if out_channels is None:
            out_channels = channels
        assert height == width  # Currently square patches only, rectangular planned
        n_patches = get_npatches(width, patch_sizes)  # largest to smallest

        # for patch[i], n*size == size of patch[i-1]
        assert verify_patches(width, patch_sizes, n_patches)
        n_patches_square = [n**2 for n in n_patches]
        mixer_dimensions = list(reversed((*n_patches_square, hidden_size)))
        mixer_widths = list(reversed((*patch_mlp_widths, hidden_mlp_width)))

        inkey, outkey, mixer_key = jr.split(key, 3)

        self.rearrange_in = lambda img: multi_patch_rearrange(
            img, n_patches, patch_sizes
        )
        self.rearrange_out = lambda img: reverse_multi_patch_rearrange(
            img, n_patches, patch_sizes
        )
        self.projection_in = eqx.nn.Linear(
            channels * patch_sizes[-1] ** 2, hidden_size, key=inkey
        )
        self.projection_out = eqx.nn.Linear(
            hidden_size, out_channels * patch_sizes[-1] ** 2, key=outkey
        )
        self.mixer = MultiMixer(
            mixer_dimensions,
            mixer_widths,
            num_blocks,
            dims_per_block=dims_per_block,
            key=mixer_key,
        )

    def __call__(self, y):
        y = self.rearrange_in(y)
        y = antivmap(self.projection_in)(y)  # nested jax.vmap
        y = self.mixer(y)
        y = antivmap(self.projection_out)(y)
        return self.rearrange_out(y)
