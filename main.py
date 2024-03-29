"""
Barry Yu
Jan 15, 2024
Pytorch DCGAN
"""

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from model import weights_init, Generator, Discriminator

# go to image number 00

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this size
image_size = 64

# Number of training epochs
num_epochs = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# We can use an image folder dataset the way we have it setup.
# Transform images to create the dataset
dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.2),
        #transforms.ElasticTransform(alpha=100.0),
        #transforms.RandomInvert(p=0.3),
        #transforms.RandomSolarize(threshold=0.9, p=0.3),
    ])
)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
'''
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()
'''


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Uncomment if starting out: apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.02.
# netG.apply(weights_init)
try:
    netG.load_state_dict(torch.load("Models/netG.pkl"))  # load netG weights
except:
    netG.apply(weights_init)

# Print the model
print(netG)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Uncomment if starting out: apply the ``weights_init`` function to randomly initialize all weights
# netD.apply(weights_init)
try:
    netD.load_state_dict(torch.load("Models/netD.pkl"))  # load netG weights
except:
    netD.apply(weights_init)

# Print the model
print(netD)


# ========== TRAINING ==========

# Define lots of stuff

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.0
fake_label = 0.0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        # some filter for if i want to train on only part of the dataset
        if (i%1 != 0): continue

        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # D() is 1 -- real image
        # D() is 0 -- fake image
        # make sure D(x) is bigger (real) and D(G(z)) is smaller (fake)
        # tldr: maximize ability to distinguish real and fake
        ###########################

        
        ## Train with real images
        
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu).view(-1)  # Forward pass real batch through D

        
        errD_real = criterion(output, label)  # Calculate loss on all-real batch
        
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()  # only used for display



        ## Train with fake images
        
        # Generate batch of latent vectors (noise
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()  # only used for display

        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake  # D(x) + D(G(z))
        # Update D
        optimizerD.step()

        
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        # D() is bigger -- real
        # objective: trick D() into thinking G(z) is real -> maximize D(G(z))
        # tldr: maximize realisticness of generated images
        ###########################
        
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Save G's output in img_list so we can see the cool animation afterwards
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


torch.save(netD.state_dict(), "Models/netD.pkl")
torch.save(netG.state_dict(), "Models/netG.pkl")

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
HTML(ani.to_jshtml())
