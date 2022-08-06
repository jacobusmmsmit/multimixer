import itertools

import jax
import jax.random as jr

import equinox as eqx


class MultiscaleMixerBlock(eqx.Module):
    mixers: list
    norms: list

    def __init__(self, dim_sizes, width_sizes, *, key):
        keys = jr.split(key, len(dim_sizes))
        self.mixers = [
            eqx.nn.MLP(dim_size, dim_size, width_size, depth=1, key=key)
            for dim_size, width_size, key in zip(dim_sizes, width_sizes, keys)
        ]
        self.norms = [eqx.nn.LayerNorm(list(reversed(dim_sizes))) for _ in dim_sizes]

    def __call__(self, y):
        N = len(self.mixers)
        i = N - 1
        # apply -i'th mixer/norm over i'th dimension
        for mixer, norm in zip(self.mixers, self.norms):
            f = mixer
            # vmap over j'th dimension for all but i'th dimension
            for j in itertools.chain(range(i), range(i + 1, N)):
                f = jax.vmap(f, j, j)
            y = y + f(norm(y))
            i -= 1
        return y
