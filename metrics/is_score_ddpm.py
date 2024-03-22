import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import pairwise_distances
import random
import glob


# Load InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=True)


def preprocess_images(img_paths):
    imgs = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        imgs.append(img)
    return np.vstack(imgs)


def calculate_is(images, n_split=10):
    preds = inception_model.predict(images)
    splits = np.array_split(preds, n_split)
    scores = []
    for preds in splits:
        p_yx = preds.mean(axis=0)
        kl_d = preds * (np.log(preds + 1e-10) - np.log(np.expand_dims(preds.mean(axis=0) + 1e-10, 0)))
        kl_d = kl_d.sum(axis=1)
        kl_d = np.mean(np.exp(kl_d))
        scores.append(kl_d)
        
    return np.mean(scores), np.std(scores)

# Load images


original_imgs = glob.glob('./DDPM_Original/*.png')
generated_imgs = glob.glob('./DDPM_Generated/*.png')


original_imgs = glob.glob('./DDPM/original_images/*.png')
ddpm_200 = glob.glob('./DDPM/200/*.png')
ddpm_400 = glob.glob('./DDPM/400/*.png')
ddpm_600 = glob.glob('./DDPM/600/*.png')
ddpm_800 = glob.glob('./DDPM/800/*.png')



images_orig_set = random.sample(original_imgs,500)
images_200_set = random.sample(ddpm_200,500)
images_400_set = random.sample(ddpm_400,500)
images_600_set = random.sample(ddpm_600,500)
images_800_set = random.sample(ddpm_800,500)



# Preprocess images
images_orig = preprocess_images(images_orig_set)
images_200 = preprocess_images(images_200_set)
images_400 = preprocess_images(images_400_set)
images_600 = preprocess_images(images_600_set)
images_800 = preprocess_images(images_800_set)

# Calculate Inception Score
is_mean_orig, is_std_orig = calculate_is(images_orig)
is_mean_200, is_std_200 = calculate_is(images_200)
is_mean_400, is_std_400 = calculate_is(images_400)
is_mean_600, is_std_600 = calculate_is(images_600)
is_mean_800, is_std_800 = calculate_is(images_800)




print("Orginal Images - Inception Score Original : Mean = {}, Std = {}".format(is_mean_orig, is_std_orig))
print("DCGAN Images - Inception Score DDPM 200: Mean = {}, Std = {}".format(is_mean_200, is_std_200))
print("DCGAN Images - Inception Score DDPM 400: Mean = {}, Std = {}".format(is_mean_400, is_std_400))
print("DCGAN Images - Inception Score DDPM 600: Mean = {}, Std = {}".format(is_mean_600, is_std_600))
print("DCGAN Images - Inception Score DDPM 800: Mean = {}, Std = {}".format(is_mean_800, is_std_800))
