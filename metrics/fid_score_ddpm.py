import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from scipy.linalg import sqrtm
import random
import glob


# Load InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

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
        kl_d = preds * (np.log(preds) - np.log(np.expand_dims(preds.mean(axis=0), 0)))
        kl_d = kl_d.sum(axis=1)
        kl_d = np.mean(np.exp(kl_d))
        scores.append(kl_d)
    return np.mean(scores), np.std(scores)

def calculate_fid(images1, images2):
    mu1, sigma1 = np.mean(inception_model.predict(images1), axis=0), np.cov(inception_model.predict(images1).T)
    mu2, sigma2 = np.mean(inception_model.predict(images2), axis=0), np.cov(inception_model.predict(images2).T)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Load images
# original_imgs = glob.glob('./Orginal_shrinked/*.jpg')
# generated_imgs = glob.glob('./GAN_Generated/*.png')

original_imgs = glob.glob('./DDPM_Original/*.png')
generated_imgs = glob.glob('./DDPM_Generated/*.png')


original_imgs = glob.glob('./DDPM/original_images/*.png')
ddpm_200 = glob.glob('./DDPM/200/*.png')
ddpm_400 = glob.glob('./DDPM/400/*.png')
ddpm_600 = glob.glob('./DDPM/600/*.png')
ddpm_800 = glob.glob('./DDPM/800/*.png')





images_orig_set = random.sample(original_imgs,500)
# images_200_set = random.sample(ddpm_200,500)
# images_400_set = random.sample(ddpm_400,500)
# images_600_set = random.sample(ddpm_600,500)
images_800_set = random.sample(ddpm_800,500)



# Preprocess images
images_orig = preprocess_images(images_orig_set)
# images_200 = preprocess_images(images_200_set)
# images_400 = preprocess_images(images_400_set)
# images_600 = preprocess_images(images_600_set)
images_800 = preprocess_images(images_800_set)



# Calculate FID
# fid_score_200 = calculate_fid(images_orig, images_200)
# fid_score_400 = calculate_fid(images_orig, images_400)
# fid_score_600 = calculate_fid(images_orig, images_600)
fid_score_800 = calculate_fid(images_orig, images_800)



# print("Fréchet Inception Distance - DDPM - 200: {}".format(fid_score_200))
# print("Fréchet Inception Distance - DDPM - 400: {}".format(fid_score_400))
# print("Fréchet Inception Distance - DDPM - 600: {}".format(fid_score_600))
print("Fréchet Inception Distance - DDPM - 800: {}".format(fid_score_800))
