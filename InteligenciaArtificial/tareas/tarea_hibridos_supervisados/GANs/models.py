"""
Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and there for
it should be called critic)
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator/Critic model

    Attributes
    ----------
    disc : nn.Sequential
        Discriminator/Critic model
    """

    def __init__(self, channels_img, features_d) -> None:
        """Init Discriminator/Critic model

        Args
        ----
        channels_img : int
            Number of channels in the input image
        features_d : int
            Number of features in the first layer of the discriminator
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(
        self, in_channels, out_channels, kernel_size, stride, padding
    ) -> nn.Sequential:
        """Returns layers of each discriminator block

        Args
        ----
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int or tuple
            Size of the convolving kernel
        stride : int or tuple
            Stride of the convolution
        padding : int or tuple
            Zero-padding added to both sides of the input

        Returns
        -------
        nn.Sequential
            Sequential of convolution, instance normalization, and leaky relu
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the discriminator/critic

        Args
        ----
        x : torch.Tensor
            Input image tensor

        Returns
        -------
        torch.Tensor
            Output of the discriminator/critic
        """
        return self.disc(x)


class Generator(nn.Module):
    """Generator model

    Attributes
    ----------
    net : nn.Sequential
        Generator model
    """

    def __init__(self, channels_noise, channels_img, features_g) -> None:
        """Init Generator model

        Args
        ----
        channels_noise : int
            Number of channels in the input noise
        channels_img : int
            Number of channels in the output image
        features_g : int
            Number of features in the first layer of the generator
        """
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(
        self, in_channels, out_channels, kernel_size, stride, padding
    ) -> nn.Sequential:
        """Returns layers of each generator block

        Args
        ----
        in_channels : int
            Number of channels in the input image
        out_channels : int
            Number of channels produced by the convolution
        kernel_size : int or tuple
            Size of the convolving kernel
        stride : int or tuple
            Stride of the convolution
        padding : int or tuple
            Zero-padding added to both sides of the input

        Returns
        -------
        nn.Sequential
            Sequential of transposed convolution, batch normalization, and relu
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the generator

        Args
        ----
        x : torch.Tensor
            Input noise tensor

        Returns
        -------
        torch.Tensor
            Output of the generator
        """
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    # print(gen)
    # print(disc)
    print("All tests passed!")


test()
