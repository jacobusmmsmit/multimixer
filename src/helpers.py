from typing import Callable

import jax
import jax.numpy as jnp
import einops
import equinox as eqx


@eqx.filter_jit
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


@eqx.filter_jit
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


def scan(f, init, it):
    state = init
    for x in it:
        state = f(state, x)
        yield state
