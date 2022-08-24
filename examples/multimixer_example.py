import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import einops

from multimixer import MultiMixer


def main():
    seed = 42
    key = jr.PRNGKey(seed)
    _, mixer_key = jr.split(key)

    # nimages = 100
    nchannels = 3
    image_hw = 64
    image_size = (nchannels, image_hw, image_hw)
    # images = jr.uniform(images_key, (nimages, *image_size))
    patch_sizes = [8, 4, 2]  # largest to smallest (global to local)

    # arbitrary settings
    hidden_size = 16
    mix_patch_sizes = [
        i + 1 for i in range(len(patch_sizes))
    ]  # auto-adjust to number of patch scales
    mix_hidden_size = 1
    num_blocks = 1
    out_channels = 3

    # This is purely to produce pretty pictures :)
    for num_blocks in [1, 5, 10]:
        m = MultiMixer(
            image_size,
            patch_sizes,
            hidden_size,
            mix_patch_sizes,
            mix_hidden_size,
            num_blocks,
            key=mixer_key,
            out_channels=out_channels,
        )

        image = jnp.ones(image_size)
        y = m(image)

        y = (y - np.min(y)) / np.ptp(y)
        plt.imshow(einops.rearrange(y, "c h w -> h w c"))
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"pretty_pictures/im{patch_sizes}_{num_blocks}.png")

    return y


main()
