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


# original_imgs = glob.glob('./DCGAN_Original/*.png')
# generated_imgs = glob.glob('./DCGAN_Generated/*.png')


original_imgs = glob.glob('./DCGAN/original_images/*.jpg')
DCGAN_200 = glob.glob('./DCGAN/200/*.png')
DCGAN_300 = glob.glob('./DCGAN/300/*.png')
DCGAN_400 = glob.glob('./DCGAN/400/*.png')
DCGAN_500 = glob.glob('./DCGAN/500/*.png')
DCGAN_600 = glob.glob('./DCGAN/600/*.png')
DCGAN_700 = glob.glob('./DCGAN/700/*.png')
DCGAN_800 = glob.glob('./DCGAN/800/*.png')



images_orig_set = random.sample(original_imgs,500)
images_200_set = random.sample(DCGAN_200,500)
images_300_set = random.sample(DCGAN_300,500)
images_400_set = random.sample(DCGAN_400,500)
images_500_set = random.sample(DCGAN_500,500)
images_600_set = random.sample(DCGAN_600,500)
images_700_set = random.sample(DCGAN_700,500)
images_800_set = random.sample(DCGAN_800,500)



# Preprocess images
images_orig = preprocess_images(images_orig_set)
images_200 = preprocess_images(images_200_set)
images_300 = preprocess_images(images_300_set)
images_400 = preprocess_images(images_400_set)
images_500 = preprocess_images(images_500_set)
images_600 = preprocess_images(images_600_set)
images_700 = preprocess_images(images_700_set)
images_800 = preprocess_images(images_800_set)

# Calculate Inception Score
is_mean_orig, is_std_orig = calculate_is(images_orig)
is_mean_200, is_std_200 = calculate_is(images_200)
is_mean_300, is_std_300 = calculate_is(images_300)
is_mean_400, is_std_400 = calculate_is(images_400)
is_mean_500, is_std_500 = calculate_is(images_500)
is_mean_600, is_std_600 = calculate_is(images_600)
is_mean_700, is_std_700 = calculate_is(images_700)
is_mean_800, is_std_800 = calculate_is(images_800)



print("Orginal Images - Inception Score Original : Mean = {}, Std = {}".format(is_mean_orig, is_std_orig))
print("DCGAN Images - Inception Score DCGAN 200: Mean = {}, Std = {}".format(is_mean_200, is_std_200))
print("DCGAN Images - Inception Score DCGAN 200: Mean = {}, Std = {}".format(is_mean_300, is_std_300))
print("DCGAN Images - Inception Score DCGAN 400: Mean = {}, Std = {}".format(is_mean_400, is_std_400))
print("DCGAN Images - Inception Score DCGAN 200: Mean = {}, Std = {}".format(is_mean_500, is_std_500))
print("DCGAN Images - Inception Score DCGAN 600: Mean = {}, Std = {}".format(is_mean_600, is_std_600))
print("DCGAN Images - Inception Score DCGAN 200: Mean = {}, Std = {}".format(is_mean_700, is_std_700))
print("DCGAN Images - Inception Score DCGAN 800: Mean = {}, Std = {}".format(is_mean_800, is_std_800))
