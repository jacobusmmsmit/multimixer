import jax.random as jr
from multimixer import ImageMixer


key = jr.PRNGKey(42)
uniform_key, mixer_key = jr.split(key)


def test_readme():
    # input image settings
    nchannels = 3
    image_hw = 64
    image_size = (nchannels, image_hw, image_hw)
    image = jr.uniform(uniform_key, image_size)

    # mixer settings
    patch_sizes = [8, 4, 2]  # largest to smallest (global to local)
    hidden_size = 16
    mix_patch_sizes = [4, 8, 10]  # same length as patch_sizes
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
    assert my_mixer(image).shape == (out_channels, image_hw, image_hw)


test_readme()
