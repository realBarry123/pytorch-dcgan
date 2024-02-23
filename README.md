# 🎨 pytorch-dcgan
This is a Generative Adversarial Network (GAN) built With Pytorch. It is trained on the CelebA dataset and generates creepy looking faces. 

Essentially, this creates 2 neural networks, the **generator** and the **discriminator**. The generator is responsible for converting noise into recognizable images, while the discriminator tell apart real images from the ones the generator creates. The models are trained and compete with each other, improving together. 

I am currently working on modifying it to generate images that look like faces both upside-down and rightside-up. Those changes will show up in the junk/upsidedowntest branch. 

Tutorial I'm following: [Pytorch.org](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
