import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake, device="cpu") -> torch.Tensor:
    """Calculates the gradient penalty loss for WGAN GP

    Args
    ----
    critic : nn.Module
        The critic or discriminator model
    real : torch.Tensor
        The real images
    fake : torch.Tensor
        The fake images
    device : str
        The device type

    Returns
    -------
    torch.Tensor
        The gradient penalty loss
    """
    # Random weight term for interpolation between real and fake samples
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar") -> None:
    """Saves model checkpoints

    Args
    ----
    state : dict
        The checkpoint state to be saved
    filename : str
        The filename to save the checkpoint
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc) -> None:
    """Loads model checkpoints

    Args
    ----
    checkpoint : str
        The filename to load the checkpoint
    gen : nn.Module
        The generator model
    disc : nn.Module
        The discriminator model
    """
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint["gen"])
    disc.load_state_dict(checkpoint["disc"])
