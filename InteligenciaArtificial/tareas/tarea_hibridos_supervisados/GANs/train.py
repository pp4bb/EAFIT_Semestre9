"""
Training of WGAN-GP
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from utils import gradient_penalty
from models import Discriminator, Generator, initialize_weights

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device, "AS DEVICE")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


config = {
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "image_size": IMAGE_SIZE,
    "channels_img": CHANNELS_IMG,
    "z_dim": Z_DIM,
    "num_epochs": NUM_EPOCHS,
    "features_critic": FEATURES_CRITIC,
    "features_gen": FEATURES_GEN,
    "critic_iterations": CRITIC_ITERATIONS,
    "lambda_gp": LAMBDA_GP,
}

# wandb config
run = wandb.init(project="GANs", config=config)

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# fixed noise for testing
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

step = 0

gen.train()
critic.train()

wandb.watch((gen, critic), log="all")

for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Log losses to wandb
        wandb.log({"Critic loss": loss_critic, "Generator loss": loss_gen})
        # Log the gradients of the generator and critic to wandb
        # wandb.log(
        #    {"Critic gradient": critic_real.grad, "Generator gradient": gen_fake.grad}
        # )
        # Log the gans loss to wandb
        wandb.log({"Gans loss": loss_critic + loss_gen})

        # Print losses occasionally and log the images to wandb
        if batch_idx % 10 == 0 and batch_idx > 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                # log to wandb
                wandb.log(
                    {
                        "Real": [wandb.Image(img_grid_real)],
                        "Fake": [wandb.Image(img_grid_fake)],
                    }
                )

            step += 1

# Save the trained generator
torch.save(gen.state_dict(), "gen.pth")
# Save the trained critic
torch.save(critic.state_dict(), "critic.pth")
