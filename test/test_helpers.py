import jax.numpy as jnp
import jax.random as jr

from src.helpers import (
    get_npatches,
    multi_patch_rearrange,
    reverse_multi_patch_rearrange,
)

key = jr.PRNGKey(42)
gen_key, linear_key = jr.split(key)

hw = 640  # Image height and width
c = 2  # Image number of channels
img = jr.uniform(gen_key, (c, hw, hw))

patch_sizes = [8, 4, 1]
n_patches = get_npatches(hw, patch_sizes)
hidden_size = 64
multi_patch_rearrange(img, n_patches, patch_sizes).shape
n_patches

def main():
    a = multi_patch_rearrange(img, n_patches, patch_sizes)
    b = reverse_multi_patch_rearrange(a, n_patches, patch_sizes)
    assert jnp.alltrue(img == b)


main()
