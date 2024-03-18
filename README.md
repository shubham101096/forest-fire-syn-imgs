# Generative Models for Synthetic Forest Fire Image Generation: A Comparative Study of GAN and Diffusion Models

This repository contains the implementation and evaluation components for our generative models project, specifically focusing on Deep Convolutional Generative Adversarial Networks (DCGAN) and Denoising Diffusion Probabilistic Models (DDPM).

## Repository Structure

The project is organized as follows:

/ \
├── dcgan.ipynb # Notebook for DCGAN model implementation \
├── ddpm.ipynb # Notebook for DDPM model implementation \
└── metrics/ \
├── DCGAN/ # Synthetic images from DCGAN at epochs 200,  400, ... \
└── DDPM/ # Synthetic images from DDPM at epochs 200, 400, ... \
└── Scripts # Python scripts for computing metrics \


## Development Workflow

### Preparing the Environment

1. **Google Colab Setup**: The DCGAN and DDPM notebooks, along with the Python scripts for metrics calculation, are uploaded to Google Colab for leveraging its computational resources.
   
### Dataset Configuration

1. **Dataset Upload**: The forest fire dataset ([[Forest Fire Dataset at Mendeley Data]](https://data.mendeley.com/datasets/gjmr63rz2r/1)) is uploaded to Google Drive for easy access.
2. **Notebook Configuration**: The path to the dataset on Google Drive is specified in the notebooks to ensure proper data access. Additionally, the directory paths for saving synthetic images during the sampling process are set.

### Model Training and Evaluation

1. **Training**: The notebooks are executed in Google Colab with the runtime environment set to use a T4 GPU (Runtime > Change runtime type > Hardware accelerator > T4 GPU) to handle the computational demands of model training.
2. **Synthetic Image Generation**: Synthetic images are generated and saved at various epochs (200, 400, 600, 800) for both models.
3. **Metrics Calculation**: The metrics scripts are run with specified paths to the directories containing the generated images, allowing for the calculation and comparison of metrics for DCGAN and DDPM models.

## Getting Started

To replicate this project:
1. Clone this repository to your local machine or Google Colab.
2. Ensure you have access to the forest fire dataset and configure the dataset path in the notebooks.
3. Follow the development workflow for model training and evaluation.





