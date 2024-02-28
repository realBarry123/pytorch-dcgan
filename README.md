# 🖼️👤 pytorch-dcgan
This is a Generative Adversarial Network (GAN) built With Pytorch. It is trained on the CelebA dataset and generates (often) creepy looking faces. 

Essentially, this creates 2 neural networks, the **generator** and the **discriminator**. The generator is responsible for converting noise into recognizable images, while the discriminator tries to tell apart real images (from the dataset) from the ones the generator creates. The models are trained and sort of compete with each other, improving together. 

Diagram of where everything goes: 
```

Noise ---[Generator]---> Fake Image ---[Discriminator]---> Generator Loss
                \
   Real Image ---ᐠ---[Discriminator]---> Discriminator Loss

```

## junk/upsidedowntest
This was a failed experiment where I tried to generate images that looked like faces both upside-down and rightside-up. The method was to calculate the generator's loss buy feeding its fake images in both orientations to the discriminator and taking the maximum of the two losses. It did not work because the discriminator will always gain a significant advantage, and I cannot find a way to make it work. Perhaps I can come back to this later. 


---
Tutorial I'm following: [Pytorch.org](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) <sub>Thanks for the code snippets!</sub>
