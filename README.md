# MultiMixer
A Multi(scale)-Mixer implemented in Equinox and an example of it as a drop in replacement for a conventional MLP-Mixer.

This package exports the Equinox module `MultiMixer`. Below is an example usage, more examples can be found in `examples/`.
```
import jax.random as jr
import jax.numpy as jnp

from multimixer import MultiMixer


def main():
    seed = 42
    key = jr.PRNGKey(seed)
    uniform_key, mixer_key = jr.split(key)

    nchannels = 3
    image_hw = 64
    image_size = (nchannels, image_hw, image_hw)
    patch_sizes = [8, 4, 2]  # largest to smallest (global to local)

    # arbitrary settings
    hidden_size = 16
    mix_patch_sizes = [4, 8, 10] # same length as patch_sizes 
    mix_hidden_size = 1
    num_blocks = 1
    out_channels = 3

    my_mixer = MultiMixer(
        image_size,
        patch_sizes,
        hidden_size,
        mix_patch_sizes,
        mix_hidden_size,
        num_blocks,
        key=mixer_key,
        out_channels=out_channels,
    )

    image = jr.uniform(uniform_key, image_size)
    y = my_mixer(image)
    return y


main()

```
