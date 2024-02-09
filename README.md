# 🎨 pytorch-dcgan
This is a Generative Adversarial Network (GAN) built With Pytorch. It is trained on the CelebA dataset and generates creepy looking faces. 

Essentially, this creates 2 neural networks, the **generator** and the **discriminator**. The generator is responsible for converting noise into recognizable images, while the discriminator tell apart real images from the ones the generator creates. The models are trained and compete with each other, improving together. 

Tutorial I'm following: [Pytorch.org](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
