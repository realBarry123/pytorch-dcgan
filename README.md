# 🎨 pytorch-dcgan
This is a Generative Adversarial Network (GAN) built With Pytorch. It is trained on the CelebA dataset and generates creepy looking faces. 

Essentially, this creates 2 neural networks, the **generator** and the **discriminator**. The generator is responsible for converting noise into recognizable images, while the discriminator tell apart real images from the ones the generator creates. The models are trained and compete with each other, improving together. 

## junk/upsidedowntest
I am currently working on modifying it to generate images that look like faces both upside-down and rightside-up. Those changes will show up in the junk/upsidedowntest branch. 

I modify the loss of the discriminator's predictions on fake images by calculating the max of the losses of normal and upside-down versions. That way the generator is encouraged to create images that satisfy the discriminator in both orientations. 

---
Tutorial I'm following: [Pytorch.org](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
