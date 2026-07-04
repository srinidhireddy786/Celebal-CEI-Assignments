# Denoising Autoencoder using MNIST

## Project Overview
This project implements a Convolutional Denoising Autoencoder using PyTorch to remove Gaussian noise from handwritten digit images from the MNIST dataset.

## Features
- Load and preprocess MNIST dataset
- Add Gaussian noise
- Build a Convolutional Denoising Autoencoder
- Train using MSE Loss and Adam Optimizer
- Generate denoised images
- Compare Original, Noisy, and Reconstructed images

## Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib

## Results
The trained autoencoder successfully removes most of the Gaussian noise while preserving the digit structure.

## Files
- `Denoising_Autoencoder_MNIST.ipynb` – Complete implementation
- `denoising_autoencoder.pth` – Trained model
