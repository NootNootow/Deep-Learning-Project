import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy import stats
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

def model(name):

    base_model = None

    if(name == 'vgg'):
        base_model = tf.keras.applications.VGG19(input_shape=(224, 244, 3),
                                               include_top=True,
                                               weights='imagenet')

    if(name == 'alexnet'):
        base_model = tf.keras.applications.Resnet(input_shape=(224, 244, 3),
                                               include_top=True,
                                               weights='imagenet')

    base_model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate= 0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return base_model

def extraction_model(name):
    base_model = None

    if(name == 'vgg'):
        base_model = tf.keras.applications.VGG19(input_shape=(224, 244, 3),
                                               include_top=False,
                                               weights='imagenet')

    if(name == 'alexnet'):
        base_model = tf.keras.applications.Resnet(input_shape=(224, 244, 3),
                                               include_top=False,
                                               weights='imagenet')
    
    return base_model

def feature_extraction(model, images):
    base_model = None
    if(model == 'vgg'):
        base_model = extraction_model(model)
    else:
        base_model = extraction_model(model)
    img = image.load_img(images)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)

    return features

def image_feature_creation(model, folders):
    
    for x in folders:
        features = feature_extraction(model, folders):
        return features

def data_transformer():
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

    return data_transformer

def regression(data):
    alphas = np.logspace(-1, 2, 5, 7)
    best_score = 0

    for alpha in alphas:
        kFold = KFold(n_splilts = 10, shuffle = True)
        
        for train, test in kFold.split(data):
            regression = Ridge()

            train = data[train]
            test = data[test]
            x_train = train[:, :-1]
            y_train = train[:, -1]
            x_test = test[:, :-1]
            y_test = test[:, -1]

            pca = PCA(n_components=100)
            x_train = pca.fit(x_train)
            x_test = pca.fit(x_test)

            regression.fit(x_train, y_train)
            y_new = regression.predict(x_test)
            score = stats.pearsonr(y_test, y_new)

            if(score > best_score):
                best_score = score
   
    return best_score

def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()