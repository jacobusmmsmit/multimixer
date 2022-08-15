import jax.random as jr
from jax import vmap
import equinox as eqx
import einops

from src.multiscalemixer import MultiMixerBlock


def multi_patch_rearrange(tensor, n_patches, patch_sizes):
    """
    tensor is of size (channels, width, height), leaves channel first, patches from large to small
    """
    temp = tensor
    for n, size in zip(n_patches, patch_sizes):
        temp = einops.rearrange(
            temp, "... (h hp) (w wp) -> ... (h w) hp wp", h=n, w=n, hp=size, wp=size
        )
    return einops.rearrange(temp, "... hp wp -> ... (hp wp)")


def reverse_multi_patch_rearrange(tensor, n_patches, patch_sizes):
    temp = einops.rearrange(
        tensor, "... (hp wp) -> ... hp wp", hp=patch_sizes[-1], wp=patch_sizes[-1]
    )
    for n, size in reversed(list(zip(n_patches, patch_sizes))):
        temp = einops.rearrange(
            temp, "... (h w) hp wp -> ... (h hp) (w wp)", h=n, w=n, hp=size, wp=size
        )
    return temp


def get_npatches(image_size, patch_sizes):
    sizes = (image_size, *patch_sizes)
    return [sizes[i] // sizes[i + 1] for i in range(len(sizes) - 1)]


def verify_patches(image_size, patch_sizes, n_patches):
    """
    asserts that the current patch_size * n_patches is the size of the previous patch_size (or width)
    e.g. for a 32 by 32 image with patch_sizes [8, 2] => we assert n_patches == [32//8 = 4, 8//2 = 4]
    """
    patches_ok = True
    last_size = image_size
    for s, n in zip(patch_sizes, n_patches):
        patches_ok = patches_ok and (s * n == last_size)
        last_size = s
    return patches_ok


class Mixer(eqx.Module):
    img_size: list
    n_patches: list
    patch_sizes: list
    hidden_size: int
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    blocks: list
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        img_size,
        patch_sizes,
        hidden_size,
        mix_patch_sizes,
        mix_hidden_size,
        num_blocks,
        *,
        key,
    ):
        channels, height, width = img_size
        assert height == width  # Currently square patches only, rectangular planned
        n_patches = get_npatches(width, patch_sizes)  # largest to smallest

        # for patch[i], n*size == size of patch[i-1]
        assert verify_patches(width, patch_sizes, n_patches)
        n_patches_square = [n**2 for n in n_patches]

        inkey, outkey, *bkeys = jr.split(key, 2 + num_blocks)

        self.img_size = img_size
        self.n_patches = n_patches
        self.patch_sizes = patch_sizes
        self.hidden_size = hidden_size
        self.linear_in = eqx.nn.Linear(channels, hidden_size, key=inkey)
        self.linear_out = eqx.nn.Linear(hidden_size, channels, key=outkey)
        self.blocks = [
            MultiMixerBlock(
                (*n_patches_square, patch_sizes[-1] ** 2 * hidden_size),
                (*mix_patch_sizes, mix_hidden_size),
                key=bkey,
            )
            for bkey in bkeys
        ]
        self.norm = eqx.nn.LayerNorm(
            (*n_patches_square, patch_sizes[-1] ** 2 * hidden_size)
        )

    def __call__(self, y):
        assert y.shape == self.img_size
        y = vmap(vmap(self.linear_in, 1, 1), 2, 2)(y)
        y = multi_patch_rearrange(y, self.n_patches, self.patch_sizes)
        y = einops.rearrange(y, "c ... p -> ... (p c)", c=self.hidden_size)

        y = self.norm(y)
        for block in self.blocks:
            y = block(y)

        y = einops.rearrange(y, "... (p c) -> c ... p", c=self.hidden_size)
        y = reverse_multi_patch_rearrange(y, self.n_patches, self.patch_sizes)
        return vmap(vmap(self.linear_out, 1, 1), 2, 2)(y)
