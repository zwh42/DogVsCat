# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:10:20 2017

@author: wenhao zhao
"""

import numpy as np
import os
import glob
import cv2
from collections import Counter
import random

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, ELU
from keras.layers.convolutional import Cropping2D, Convolution2D, MaxPooling2D
from keras.layers.core import Lambda, Dense, Activation, Flatten

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)

def resize_image(image, shape = (500, 500)):
    resized_image = cv2.resize(image, shape)
    return resized_image


def label_one_hot_encoding(labe_list):
    encoder = LabelEncoder()
    transfomed_label = encoder.fit_transform(labe_list)
    one_hot_encoded_labels =  np_utils.to_categorical(transfomed_label)
    one_hot_label_dict = {}
    for i  in range(len(labe_list)):
        one_hot_label_dict[labe_list[i]] = one_hot_encoded_labels[i] 
    
    print("one hot encoding: ", one_hot_label_dict)
    return one_hot_label_dict


def load_samples(DATA_PATH, one_hot_encoding_dict):
    labels = one_hot_encoding_dict.keys()    
    samples = [] # [one_hot_label, image_path]
    temp_size = 0
    for dir in DATA_PATH: 
        print("loading data from: " + dir)
        for key in labels:           
            samples += [[one_hot_encoding_dict[key],file] for file in glob.glob(os.path.join(dir, key + '*.jpg'))]
            temp_size = len(samples) - temp_size    
            print("total " + str(temp_size) + " " + key + " images loaded.")
            temp_size = len(samples)
        
        print("total " + str(temp_size) + " raw data samples loaded.")
        
    return samples  


def raw_data_analysis(raw_sample_list, one_hot_encoding_dict):
    
    inverse_one_hot_encoding_dict = {tuple(v): k for k, v in one_hot_encoding_dict.items()}
    
    print(inverse_one_hot_encoding_dict)
    
    image_size_counter = {}   
    dimension_shape = ["height", "width"]
    
    for sample in raw_sample_list:   
        label = inverse_one_hot_encoding_dict[tuple(sample[0])]
        if label  not in image_size_counter:
            image_size_counter[label]= {}
            for i in range(len(dimension_shape)):
                image_size_counter[label][dimension_shape[i]] = Counter() 
        
        img = cv2.imread(sample[1])
        for i in range(len(dimension_shape)):
            image_size_counter[label][dimension_shape[i]][img.shape[i]] += 1

            
    for label in image_size_counter.keys():
        for dim in dimension_shape:
             print("most common " + label + " image " + dim + ": " , image_size_counter[label][dim].most_common(10))
            

    return image_size_counter


def sample_generator(samples, batch_size=10):
    
    while True:
        samples = shuffle(samples)
        num_samples = len(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            one_hot_labels = []
            
            for sample in batch_samples:
                image = cv2.imread(sample[1])
                image = resize_image(image)

                images.append(image)
                one_hot_labels.append(sample[0])
                #plt.imshow(image)
                #plt.show()
            
            X_train = np.array(images)
            y_train = np.array(one_hot_labels)
            
            #print("y_train", y_train)
            yield shuffle(X_train, y_train)



def model_setup(input_shape = (500, 500, 3), num_classes = 2):
    model = Sequential()
    model.add(Convolution2D(32,3,3, dim_ordering='tf', input_shape = input_shape))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(32,3,3))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Convolution2D(64,3,3))    
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))    
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss = 'categorical_crossentropy',  optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    return model


def flow_setup():
    
    one_hot_encoding_dict = label_one_hot_encoding(LABEL_LIST)
    print("label encoding is done.")
    
    raw_samples = load_samples(TRAIN_DATA_PATH_LIST, one_hot_encoding_dict)
    print("raw sample loaded.")
    
    image_shape_counter = raw_data_analysis(raw_samples, one_hot_encoding_dict)
    
    
    if DO_VISUALIZE:
        visualize_image_size_distribution(image_shape_counter)
    
    train_validation_samples, test_samples = train_test_split(raw_samples, test_size = 0.2, random_state = 42)
    train_samples, validation_samples = train_test_split(train_validation_samples, test_size = 0.2, random_state = 42)
    print("train sample count: ", len(train_samples), "\nvalidation sample count: ", len(validation_samples), "\ntest sample count: ", len(test_samples))
    print("sample data example:\n", train_samples[random.randint(0, len(train_samples))])
    
    train_generator = sample_generator(train_samples, batch_size = 32)
    validation_generator = sample_generator(validation_samples, batch_size = 32)
    test_generator = sample_generator(test_samples, batch_size = 32)
    
    model = model_setup(input_shape = (500, 500, 3), num_classes = len(LABEL_LIST))
    print("start model fitting...")
    history_object = model.fit_generator(train_generator, samples_per_epoch= int(len(train_samples)), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1, verbose=1)
    score = model.evaluate_generator(test_generator, 1500, max_q_size=10, nb_worker=1, pickle_safe=False)
    print(score)



if __name__ == "__main__":

    # general setting    
    DO_VISUALIZE = False
    TRAIN_DATA_PATH_LIST = ["./train"]
    TEST_DATA_PATH_LIST = ["./test"]

    LABEL_LIST = ["dog", "cat"]
    
    flow_setup()
    
    