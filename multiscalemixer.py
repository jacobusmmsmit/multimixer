import itertools

import jax
import jax.random as jr

import equinox as eqx


class MultiscaleMixerBlock(eqx.Module):
    mixers: list
    norms: list

    def __init__(self, dimensions, mlp_widths, *, key):
        assert len(dimensions) == len(mlp_widths)
        mlp_keys = jr.split(key, len(dimensions))
        self.mixers = [
            eqx.nn.MLP(dim, dim, mlp_width, depth=1, key=mlp_key)
            for dim, mlp_width, mlp_key in reversed(
                list(zip(dimensions, mlp_widths, mlp_keys))
            )
        ]
        self.norms = [eqx.nn.LayerNorm(tuple(reversed(dimensions))) for _ in dimensions]

    def __call__(self, y):
        # TODO: improve compilation time by structured control flow primitives
        # for all i, we vmap mixer i over all other j dimensions
        N = len(self.mixers)
        for i, (mixer, norm) in enumerate(zip(self.mixers, self.norms)):
            f = mixer
            for j in itertools.chain(range(i), range(i + 1, N)):
                f = jax.vmap(f, j, j)
            y = y + f(norm(y))
        return y
