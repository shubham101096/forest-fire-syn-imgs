import os
import random
from PIL import Image
import matplotlib.pyplot as plt


random.seed(40)

# Define the path to the main dcgan/ddpm folder
folder_path = '.'


# Function to create and save a 4x4 grid image from a list of image paths
def create_and_save_grid(image_paths, output_path):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    # Hide axes and ticks
    for ax in axes.flat:
        ax.axis('off')

    # Load and display each image in the grid
    for ax, img_path in zip(axes.flat, image_paths):
        img = Image.open(img_path)
        ax.imshow(img)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_path)
    plt.close()


# Process each sub-folder that contains images for a specific epoch
for subdir in os.listdir(folder_path):
    if subdir == 'venv' or subdir.startswith('.'):
        continue
    subdir_path = os.path.join(folder_path, subdir)
    if os.path.isdir(subdir_path):
        # List all image files in the subdirectory
        image_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if
                       os.path.isfile(os.path.join(subdir_path, f))]

        # Randomly select 16 images
        selected_images = random.sample(image_files, min(16, len(image_files)))

        # Create and save the 4x4 grid image
        output_path = os.path.join(folder_path, f"dcgan_{subdir}epoch_grid.png")
        create_and_save_grid(selected_images, output_path)
        print(f"Grid image saved to {output_path}")
