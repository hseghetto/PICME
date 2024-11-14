# PICME

Scientific initiation (Nov 2019 - Aug 2021) in Deep Learning at PUC-Rio, supervised by Dr. Matias G Delgadino as part of PICME.

The repository includes .ipynb files with implementations in tensorflow and keras created as a study of different types of models. All models were trained using the MNIST handwritten digits dataset.

## Conditional GAN

This is a simplified implementation of the network described in Mirza and Osindero (2014), which adapts the GAN framework introduced by Goodfellow et al. (2014). The key modification is the inclusion of a second input for the labels associated with each image type, enabling the generation of class-conditioned images.

Results after 25 epochs:
![Results afetr 25 epochs](https://github.com/user-attachments/assets/43345dac-66d2-4630-93bb-9a3a23ec130e)

Results after 50 epochs:
![Results after 50 epochs](https://github.com/user-attachments/assets/82aac72c-b4e1-4009-9d48-1ce47913789c)

Training progress GIF:
![cgan](https://github.com/user-attachments/assets/71d428ae-4869-4c31-9d2b-0b76b23f51eb)
The model was trained for a total of 50 epochs, with each epoch taking approximately 30 seconds. The estimated total training time was around 25 minutes. This includes image generation after each epoch.

## Progressive Growing GAN

This implementation is a simplified version of the network described in Karras et al. (2017). The approach begins with smaller generator and discriminator models that produce low-resolution images, enabling faster training. These models can later be expanded to generate higher-resolution images. This strategy is expected to achieve the final results more quickly and with greater stability compared to starting with a model that generates high-resolution images from scratch.

Training progress GIFs (real size and upscaled):
![nn_1_real](https://github.com/user-attachments/assets/d3faedac-8d25-47df-81cc-8d337491c22d)
![nn_1_scaled](https://github.com/user-attachments/assets/eb54e33b-8e91-45fa-a15d-9c4ff4699d95)


## Conditional Growing GAN

This network combines elements of the Conditional GAN and Progressive Growing GAN (PG-GAN) as a proof of concept for a system that can both generate images at increasing resolutions and control the class of the generated images. The generator and discriminator are designed with self-similar architectures, regardless of model size. This approach uses PG-GAN as the foundation, incorporating modifications to include a label head in the input blocks of both models. The most significant change was to the discriminator, as its input dimensions shift with each expansion. An alternative approach could have been to only expand the image head of the discriminator, but this would result in an architecture that is no longer self-similar after scaling.

Training progress GIFs (real size and upscaled):
![nn_1_real](https://github.com/user-attachments/assets/10dd139b-9f4d-4bd0-a340-cb98a652b3ab)
![nn_1_scaled](https://github.com/user-attachments/assets/929c75b6-23d1-489c-8e0f-ff54b45ba885)

