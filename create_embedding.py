from os import listdir
from pickle import dump
import keras
import glob
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tqdm import tqdm

#
def extract_features(directory,target_size):
    model = ResNet50()
    #Modify model to remove the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
    print(model.summary())
    
    # extracting feature of real
    real_img = glob.glob(f"{directory}/test/real/**/*.png")
    features=[]
    for img_name in tqdm(real_img):
        image = load_img(img_name,target_size=(target_size,target_size))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        img_feature = model.predict(image, verbose=0)
        # store feature
        features.append(img_feature)
    dump(features, open("REAL_TEST.pkl", 'wb'))

    # extracting feature of spoof/fake
    fake_img = glob.glob(f"{directory}/test/spoof/**/*.png")
    features = []
    for img_name in tqdm(fake_img):
        image = load_img(img_name, target_size=(target_size, target_size))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        img_feature = model.predict(image, verbose=0)
        # store feature
        features.append(img_feature)
    dump(features, open("SPOOF_TEST.pkl", 'wb'))
    
    # extracting feature of real
    real_img = glob.glob(f"{directory}/train/real/**/*.png")
    features=[]
    for img_name in tqdm(real_img):
        image = load_img(img_name,target_size=(target_size,target_size))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        img_feature = model.predict(image, verbose=0)
        # store feature
        features.append(img_feature)
    dump(features, open("REAL_TRAIN.pkl", 'wb'))

    # extracting feature of spoof/fake
    fake_img = glob.glob(f"{directory}/train/spoof/**/*.png")
    features = []
    for img_name in tqdm(fake_img):
        image = load_img(img_name, target_size=(target_size, target_size))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        img_feature = model.predict(image, verbose=0)
        # store feature
        features.append(img_feature)
    dump(features, open("SPOOF_TRAIN.pkl", 'wb'))
    
extract_features("dataset",224)