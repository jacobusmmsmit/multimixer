from typing import Callable

import einops
import equinox as eqx
import jax
import jax.numpy as jnp


# Rearrange needs patch_sizes[-1] then reversed npatches[1:]
# the reverse operation additionally needs npatches[0].
# Hence, for now, I'm leaving the rearranges defined explicitly in terms of
# patches on images. In the future we could make this rearranging input agnostic.
@eqx.filter_jit
def multi_patch_rearrange(tensor, n_patches, patch_sizes):
    """
    tensor is of size (channels, width, height), leaves channel first, patches from large to small
    """
    temp = einops.rearrange(
        tensor,
        "c (Hh ph) (Ww pw) -> (c ph pw) Hh Ww",
        ph=patch_sizes[-1],
        pw=patch_sizes[-1],
    )
    for n in reversed(n_patches[1:]):
        temp = einops.rearrange(temp, "... (H h) (W w) -> ... (h w) H W", w=n, h=n)
    return einops.rearrange(temp, "... H W -> ... (H W)")


@eqx.filter_jit
def reverse_multi_patch_rearrange(tensor, n_patches, patch_sizes):
    """
    reverses [`multi_patch_rearrange`]
    """
    temp = einops.rearrange(
        tensor, "... (H W) -> ... H W", H=n_patches[0], W=n_patches[0]
    )
    for n in n_patches[1:]:
        temp = einops.rearrange(temp, "... (h w) H W -> ... (H h) (W w)", w=n, h=n)
    temp = einops.rearrange(
        temp,
        "(c ph pw) Hh Ww -> c (Hh ph) (Ww pw)",
        ph=patch_sizes[-1],
        pw=patch_sizes[-1],
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


def antivmap(fn: Callable, axis: int = 0) -> Callable:
    """Returns a function which `vmap`s `fn` over all axes of its input except `axis`.

    Example:
    ```
    import jax.numpy as jnp
    import jax.random as jr
    from equinox.nn import MLP

    key = jr.PRNGKey(42)
    unif_key, mlp_key = jr.split(key)
    A = jr.uniform(unif_key, (10, 10, 5))
    my_mlp = MLP(10, 10, width_size=20, depth=2, key=mlp_key)
    antivmap(my_mlp)(A)
    ```
    """

    def wrapped_fn(x):
        op = fn
        for i in range(jnp.ndim(x)):
            if i == axis:
                continue
            op = jax.vmap(op, in_axes=i, out_axes=i)
        return op(x)

    return wrapped_fn
