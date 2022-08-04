import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.nn as jnn
import equinox as eqx
from equinox.nn import LayerNorm, MLP

from jax import vmap, jit, grad
from einops import rearrange, reduce, repeat

import matplotlib
import matplotlib.pyplot as plt


def cyclic_permutations(a):
    permutations = [a[idx:] + a[:idx] for idx in reversed(range(len(a)))]
    return permutations[-1:] + permutations[:-1]


### MLP-Mixer:
# Visually
# input is of size (M, N) Channels by Patches
# input -> eqx.nn.LayerNorm -> (M, N) => (N, M) -> MLP1 -> (N, M) => (M, N) ->+ -> O -> LayerNorm -> MLP2 -> + -> output
#  V                                                                          ^    V                         ^
# skip-connection ----------------------------->------------------------------^    skip-connection ----->----^

# In Factory, Module, Op notation
# Setup: M, N, DS, DC = 16, 3, 256, 2048
# MLP1 ~ MLP([M, DS, DS, M], gelu)
# MLP2 ~ MLP([N, DC, DC, N], gelu)
# LN ~ LayerNorm()
# Tr ~ Transpose()
# Leading to the two components
# f: x -> (TR.vmap(MLP1).TR.LN)(x) + x
# g: x -> (vmap(MLP2).LN)(x) + x
# MLPMixer(x) = (g.f)(x)


def accumulate(f, x):
    """
    A skip connection
    """
    return f(x) + x


def ps(obj):
    """
    Utility function for printing the shape of an object
    """
    print(jnp.shape(obj))


class MLPMixer(eqx.Module):
    MLP1: eqx.nn.MLP
    MLP2: eqx.nn.MLP
    LN1: eqx.nn.LayerNorm
    LN2: eqx.nn.LayerNorm

    def __init__(self, nchannels, npatches, hidden_width1, hidden_width2, key):
        mlp1_key, mlp2_key = jrd.split(key)
        self.MLP1 = MLP(nchannels, nchannels, hidden_width1, 2, jnn.gelu, key=mlp1_key)
        self.MLP2 = MLP(npatches, npatches, hidden_width2, 2, jnn.gelu, key=mlp2_key)
        self.LN1 = LayerNorm(None, elementwise_affine=False)
        self.LN2 = LayerNorm(None, elementwise_affine=False)

    def __call__(self, x):
        """
        (Residual NoMix) ∘ (Residual Mix)
        """

        def mix(x):
            """
            mix = Tr ∘ vmap(MLP1) ∘ Tr ∘ LN1
            """
            z0 = self.LN1(x)
            z1 = rearrange(z0, "c p -> p c")
            z2 = vmap(self.MLP1)(z1)
            z3 = rearrange(z2, "p c -> c p")
            return z3

        def nomix(x):
            """
            nomix = vmap(MLP2) ∘ LN1
            """
            z0 = self.LN2(x)
            z1 = vmap(self.MLP2)(z0)
            return z1

        z1 = accumulate(mix, x)
        y = accumulate(nomix, z1)
        return y


if __name__ == "__main__":
    key = jrd.PRNGKey(40)

    def main1():
        nchannels = 3
        npatches = 16
        hidden_width1 = 16
        hidden_width2 = 32

        Mix1 = MLPMixer(nchannels, npatches, hidden_width1, hidden_width2, key)
        tkn = jrd.uniform(key, (nchannels, npatches))
        print(Mix1(tkn))

    main1()

    # New stuff:
    mlp1_key, mlp2_key = jrd.split(key)
    imsize = 16, 16, 3
    height, width, channels = imsize
    # Given an image, which is an array of shape (height, width, channels),
    img = rearrange(
        vmap(lambda x: jnp.reshape(jnp.repeat(x, height * width), (height, width)))(
            jnp.arange(channels)
        ),
        "c h w -> h w c",
    )
    # the idea is to reshape this according to `(h p) (w q) c -> (h w) (p q c)` (using Einops notation).
    reshaped = rearrange(img, "(h nh) (w nw) c -> (h w) (nh nw c)", nh=2, nw=2)

    # Then apply an MLP down one dimension (vmap'ing over the other),
    n_glob, n_loc = jnp.shape(reshaped)
    ## mix = Tr ∘ vmap(MLP1) ∘ Tr ∘ LN1
    MLP1 = MLP(n_glob, n_glob, 4, 2, jnn.gelu, key=mlp1_key)
    LN1 = LayerNorm(None, elementwise_affine=False)

    z0 = LN1(reshaped)
    z1 = rearrange(z0, "c p -> p c")
    z2 = vmap(MLP1)(z1)
    z3 = rearrange(z2, "p c -> c p")
    z4 = z3 + reshaped

    # followed by an MLP down the other dimension (vmap'ing over the first).
    MLP2 = MLP(n_loc, n_loc, 4, 2, jnn.gelu, key=mlp2_key)
    LN2 = LayerNorm(None, elementwise_affine=False)

    y0 = LN2(z4)
    y1 = vmap(MLP2)(y0)
    y2 = y1 + z4
    ## Output of this:
    # h = 64//2
    # w = 64//2
    # unshaped = rearrange(y2, '(h w) (nh nw c) -> (h nh) (w nw) c', h = 32, w = 32, c = channels, nw = 2)
    # new_img = (unshaped - jnp.min(unshaped)) / jnp.max(unshaped - jnp.min(unshaped))

    # plt.imshow(new_img, interpolation='nearest')
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("hm.png")

    # Instead of reshaping into a two-dimensional (patches, channels) array, suppose we were to reshape
    # into some three-dimensional (?, ?, ?) array, or a four-dimensional (?, ?, ?, ?) array?
    # What would that look like / what would be meaningful?
    # I believe that the appropriate extension point here is wrt the local/nonlocal distinction.
    # For a three dimensional array, this would mean axes corresponding to "small scale",
    # "medium scale" "large scale" information respectively.
    # And indeed we can express this concisely in Einops notation very easily:
    # `(h p a) (w q b) c -> (h w) (p q) (c a b)` would be the three-dimensional analogue;
    # `(h p a x) (w q b y) c -> (h w) (p q) (a b) (c x y)` would be the four-dimensional analogue; etc.
    threeshaped = rearrange(
        img,
        "(h nh1 nh2) (w nw1 nw2) c -> (h w) (nh1 nw1) (nh2 nw2 c)",
        nh1=2,
        nw1=2,
        nh2=2,
        nw2=2,
    )
    # We would then apply an MLP down one of these dimensions, whilst vmap'ing over all of the others.
    # Then repeat, working our way through each dimension in turn. (As with the alternation between dimensions in the 2D case.)
    macromlp_key, mesomlp_key, micromlp_key = jrd.split(key, 3)
    n_macro, n_meso, n_micro = jnp.shape(threeshaped)
    MicroMLP = MLP(n_micro, n_micro, 32, 2, jnn.gelu, key=micromlp_key)
    MesoMLP = MLP(n_meso, n_meso, 32, 2, jnn.gelu, key=mesomlp_key)
    MacroMLP = MLP(n_macro, n_macro, 32, 2, jnn.gelu, key=macromlp_key)
    MicroLN = LayerNorm(None, elementwise_affine=False)
    MesoLN = LayerNorm(None, elementwise_affine=False)
    MacroLN = LayerNorm(None, elementwise_affine=False)


class MixerBlock3(eqx.Module):
    micro_mixer: eqx.nn.MLP
    meso_mixer: eqx.nn.MLP
    macro_mixer: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm

    def __init__(self, dim_sizes, width_sizes, *, key):
        macro_key, meso_key, micro_key = jrd.split(key, 3)
        micro_size, meso_size, macro_size = dim_sizes
        micro_width_size, meso_width_size, macro_width_size = width_sizes
        self.micro_mixer = eqx.nn.MLP(
            micro_size, micro_size, micro_width_size, depth=1, key=micro_key
        )
        self.meso_mixer = eqx.nn.MLP(
            meso_size, meso_size, meso_width_size, depth=1, key=meso_key
        )
        self.macro_mixer = eqx.nn.MLP(
            macro_size, macro_size, macro_width_size, depth=1, key=macro_key
        )
        self.norm1 = eqx.nn.LayerNorm((macro_size, meso_size, micro_size))
        self.norm2 = eqx.nn.LayerNorm((micro_size, macro_size, meso_size))
        self.norm3 = eqx.nn.LayerNorm((meso_size, micro_size, macro_size))

    def __call__(self, y):
        y = vmap(vmap(MicroMLP))(MicroLN(y)) + y
        y = rearrange(y, "macro meso micro -> micro macro meso")
        y = vmap(vmap(MesoMLP))(MesoLN(y)) + y
        y = rearrange(y, "micro macro meso -> meso micro macro")
        y = vmap(vmap(MacroMLP))(MacroLN(y)) + y
        y = rearrange(y, "meso micro macro -> macro meso micro")
        return y


MixerBlock3([4, 4, 4], [32, 32, 16], key=key)

# We could combine these, but we lack the information for LayerNorms
class MixerUnit(eqx.Module):
    mixer: eqx.nn.MLP
    norm: eqx.nn.LayerNorm

    def __init__(self, dim_size, width_size, *, key):
        self.mixer = eqx.nn.MLP(dim_size, dim_size, width_size, depth=1, key=key)
        self.norm = eqx.nn.LayerNorm()

    def __call__(self, y):
        y = vmap(vmap(self.mixer))(self.norm(y)) + y
        y = rearrange(y, "... d  -> d ...")
        return y


# Or we could make the mixerblock a list of individual mixers and norms
class MixerBlockND(eqx.Module):
    mixers: list
    norms: list

    def __init__(self, dim_sizes, width_sizes, *, key):
        # assert len(dim_sizes) == len(width_sizes)
        keys = jrd.split(key, len(dim_sizes))
        self.mixers = [
            eqx.nn.MLP(dim_size, dim_size, width_size, depth=1, key=key)
            for dim_size, width_size, key in zip(dim_sizes, width_sizes, keys)
        ]
        self.norms = [eqx.nn.LayerNorm(dims) for dims in cyclic_permutations(dim_sizes)]
        # self.norms = [eqx.nn.LayerNorm(None, elementwise_affine=False) for _ in dim_sizes]

    def __call__(self, y):
        for mixer, norm in zip(self.mixers, self.norms):
            y = vmap(vmap(mixer))(norm(y)) + y
            y = rearrange(y, "... d  -> d ...")
        return y

    dim_sizes = [2, 4, 6, 8]
    jnp.transpose(
        dim_sizes,
    )
    cyclic_permutations(dim_sizes)
