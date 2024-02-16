
import random
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from model import weights_init, Generator, Discriminator

manualSeed = random.randint(1, 10000) # set random seed

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

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

# For if you have GPUs
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Create and load the generator
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load("Models/netG.pkl"))

# Create and load the discriminator
netD = Discriminator(ngpu).to(device)
netD.load_state_dict(torch.load("Models/netD.pkl"))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # generate noise 
img_list = []

fake = netG(fixed_noise).detach().cpu()  # generate images
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))  #add images to list

# show images
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
plt.show()


