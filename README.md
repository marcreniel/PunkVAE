# PunkVAE: Variational Autoencoder for CryptoPunks

PunkVAE is a Variational Autoencoder (VAE) designed to generate new CryptoPunks from a latent space. By learning to encode and reconstruct 128x128x3 (RGB) CryptoPunks, the model creates new and unique CryptoPunk images when decoding from the latent space.

## Project Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. To get started:

```bash
cd model
poetry install
poetry run jupyter notebook
```

## Dataset

The model is trained on the `huggingnft/cryptopunks` dataset, which consists of 128x128x3 pixel images of CryptoPunks. The dataset is split into 80% for training and 20% for testing.

## Model Architecture and Hyperparameters

### Encoder and Decoder:

- **4 Convolutional Layers (Encoder):** 
  - Layer 1: 32 filters, 3x3 kernel, ReLU activation, Max Pooling, and 38% Dropout.
  - Layer 2: 64 filters, 3x3 kernel, ReLU activation, Max Pooling, and 38% Dropout.
  - Layer 3: 128 filters, 3x3 kernel, ReLU activation, Max Pooling, and 38% Dropout.
  - Layer 4: 256 filters, 3x3 kernel, ReLU activation, Max Pooling, and 38% Dropout.
  - These layers capture and compress features from the input image, progressively reducing its dimensionality.

- **Latent Space:**
  - **Latent Dimension:** 256.
  - The latent space encodes the compressed representation of the input image, serving as the basis for generating new CryptoPunks. The VAE learns to balance compression while maintaining useful image features.

- **4 Transposed Convolutional Layers (Decoder):**
  - Layer 1: 128 filters, 3x3 kernel, ReLU activation, and 36% Dropout.
  - Layer 2: 64 filters, 3x3 kernel, ReLU activation, and 36% Dropout.
  - Layer 3: 32 filters, 3x3 kernel, ReLU activation, and 36% Dropout.
  - Layer 4: 3 filters, 3x3 kernel, Sigmoid activation for output.
  - These layers reconstruct the image from the latent representation, gradually upsampling the data to its original dimensions.

### Loss Function and Optimization:

- **MSE Loss:** Measures the pixel-wise difference between the original and reconstructed images. A lower MSE indicates more accurate reconstructions.
- **KL Divergence:** Encourages the distribution of latent variables to approximate a standard Gaussian distribution, facilitating the generation of new, diverse images. The **beta coefficient of 3.4** balances the reconstruction accuracy with the regularization effect.
- **Optimizer:**
  - **Adam:** Used for its adaptive learning rate and efficient convergence.
  - **Learning Rate:** Set to **0.0008**, chosen to balance between fast convergence and stable training.

### Training:

- **Batch Size:** 64. This batch size provides a good balance between training stability and memory efficiency. (This was all trained on an M1 Pro's Metal Performance Shaders!)
- **Epochs:** 30. This number of epochs is selected to allow the model sufficient time to learn meaningful representations without overfitting the model and/or decreasing the value of the KL.
