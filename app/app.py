import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Encoding network based on a simple forward feed neural network
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.38)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.38)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.38)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.38)
        
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc_mean = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

# Decoding network based on a simple forward feed neural network
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 256 * 8 * 8)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout1 = nn.Dropout(0.36)
        
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout(0.36)
        
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout3 = nn.Dropout(0.36)
        
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 256, 8, 8)
        
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.conv4(x))
        return x

# Define the Autoencoder Model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    # Reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Load the saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load('autoencoder_complete.pth', map_location=torch.device('cpu'))
    model.eval()
    return model

vae = load_model()

# Function to generate and display images
def generate_and_display_images(model, latent_dim, num_images=8):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim)
        generated_images = model.decoder(z)
    
    # Convert to grid and display using Matplotlib
    img_grid = make_grid(generated_images, nrow=num_images, normalize=True)
    np_img = img_grid.permute(1, 2, 0).numpy()
    plt.imshow(np_img)
    plt.axis('off')
    st.pyplot()

# Streamlit app layout
st.title("CryptoPunks VAE Generator")
st.write("Generate new CryptoPunks by sampling from the latent space.")

num_images = st.slider("Number of images to generate", 1, 16, 8)
if st.button("Generate"):
    generate_and_display_images(vae, latent_dim=256, num_images=num_images)
