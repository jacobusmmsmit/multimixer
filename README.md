# MultiMixer
A Multi(scale) MLP-Mixer module for [Equinox](https://www.github.com/patrick-kidger/equinox).

This package exports the Equinox modules `MultiMixer` and `ImageMixer`. Below is an example usage, more examples can be found in `examples/`.
```python
import jax.random as jr
import jax.numpy as jnp

from multimixer import ImageMixer

key = jr.PRNGKey(42)
uniform_key, mixer_key = jr.split(key)

# input image settings
nchannels = 3
image_hw = 64
image_size = (nchannels, image_hw, image_hw)
image = jr.uniform(uniform_key, image_size)

# mixer settings
patch_sizes = [8, 4, 2]  # largest to smallest (global to local)
hidden_size = 16
mix_patch_sizes = [4, 8, 10] # same length as patch_sizes 
mix_hidden_size = 1
num_blocks = 1
out_channels = 1

my_mixer = ImageMixer(
    image_size,
    patch_sizes,
    hidden_size,
    mix_patch_sizes,
    mix_hidden_size,
    num_blocks,
    key=mixer_key,
    out_channels=out_channels,
)
my_mixer(image)
```

To develop and contribute to this package run the following command to install the dependencies for development and running examples and tests:
```
pip install .[dev]
```