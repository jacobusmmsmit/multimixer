import jax
import einops

import jax.numpy as jnp

from jax import lax


def antivmap(fn, axis):
    def wrapped_fn(x):
        op = fn
        for i in range(jnp.ndim(x)):
            if i == axis:
                continue
            op = jax.vmap(op, in_axes=i, out_axes=i)
        return op(x)

    return wrapped_fn


def divscan(init, array, length=None, reverse=False, unroll=1):
    """
    Return a `floored` division scan on array, beginning with init.
    """

    def floored_div(result, element):
        result = jnp.floor_divide(result, element)
        return result, result

    return lax.scan(floored_div, init, array, length, reverse, unroll)[1]


def n_rearrange(img, num_heights, num_widths):
    """
    equivalent to einops.rearrange(img, 'c (h a?) (w b?) -> (h w) (?a ?b) ... (a b) c', a=nh, b=nw)
    """
    res = einops.rearrange(img, "c (h h0) (w w0) -> c h w")
    for nw, nh in zip(num_heights, num_widths):
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


def scan(f, init, it):
    state = init
    for x in it:
        state = f(state, x)
        yield state
