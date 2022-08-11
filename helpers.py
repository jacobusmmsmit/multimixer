import jax
import einops

import jax.numpy as jnp

from jax import lax



def divscan(init, array, length=None, reverse=False, unroll=1):
    """
    Return a `floored` division scan on array, beginning with init.
    """

    def floored_div(result, element):
        result = jnp.floor_divide(result, element)
        return result, result

    return lax.scan(floored_div, init, array, length, reverse, unroll)[1]


def arrange_patches(img, num_heights, num_widths):
    """
    equivalent to einops.rearrange(img, 'c (h a?) (w b?) -> (h w) (?a ?b) ... (c a b)', a=x, b=y)
    """
    res = einops.rearrange(
        img,
        "c (h nh) (w nw) -> h w (c nh nw)",
        nh=int(num_widths[0]),
        nw=int(num_heights[0]),
    )
    for nw, nh in zip(num_heights[1:], num_widths[1:]):
        res = einops.rearrange(
            res, "(h nh) (w nw) ... -> h w (nh nw) ...", nh=int(nh), nw=int(nw)
        )
    res = einops.rearrange(res, "h w ... -> (h w) ...")
    return res


def list_divscan(init, xs):
    """
    Not using lax.scan as it returns a tracked array and I want a list
    """
    carry = init
    ys = []
    for x in xs:
        carry //= x
        ys.append(carry)
    return jnp.stack(ys)
