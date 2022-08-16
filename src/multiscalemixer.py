import jax.random as jr
import equinox as eqx

from src.helpers import antivmap


class MultiMixerBlock(eqx.Module):
    """Maps a different MLP over each dimension of the input from last to first."""

    mixers: list
    norms: list

    def __init__(self, dimensions, mlp_widths, *, key):
        """**Arguments:**
        - `dimensions`: The dimensions of the input and output.
        - `mlp_widths`: The number of hidden layers of the MLP of each dimension.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        # dimensions are put in from global to local
        assert len(dimensions) == len(mlp_widths)
        mlp_keys = jr.split(key, len(dimensions))
        self.mixers = [
            eqx.nn.MLP(dim, dim, mlp_width, depth=1, key=mlp_key)
            for dim, mlp_width, mlp_key in zip(dimensions, mlp_widths, mlp_keys)
        ]
        self.norms = [eqx.nn.LayerNorm(dimensions) for _ in dimensions]

    def __call__(self, y):
        """**Arguments**
        - `y`: The input. Should be of shape `(dimensions)`.
        """
        # TODO: improve compilation time by structured control flow primitives
        # something like: lax.fori_loop(0, len(self.mixers), vmap(mixer[i], i, i)(norm[i]), y)
        # TODO: is this really the best way to apply the mixers in reverse? It only involves
        # changing a single line of code.
        for i, (mixer, norm) in reversed(list(enumerate(zip(self.mixers, self.norms)))):
            y = y + antivmap(mixer, i)(norm(y))
        return y
