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
from keras.layers.pooling import GlobalAveragePooling2D

from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions




import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)

def resize_image(image, output_shape = (224, 224,3)):
    resized_image = cv2.resize(image, (output_shape[0], output_shape[0]))
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
                image = resize_image(image, output_shape = IMAGE_INPUT_SHAPE)

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
    
    
    model.add(Lambda(lambda x: x/127.5 -1., input_shape = input_shape))
    model.add(Convolution2D(32,3,3))
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
    

    model.compile(loss = 'categorical_crossentropy',  optimizer="rmsprop", metrics=['accuracy'])
    print(model.summary())
    return model


def model_flow_setup(input_shape, num_classes):
    model = Sequential()
    
    
    model.add(Lambda(lambda x: x/127.5 -1., input_shape = input_shape))
    model.add(Convolution2D(32,3,3))
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
    model.add(Dense(num_classes))
    return model





def create_models(input_shape = (224, 224, 3), num_classes = 2, output_activation_list = ['sigmoid'] , loss_function_list = ['categorical_crossentropy'], optimizer_list = ["rmsprop", "adam"], metrics_list=['accuracy']):
    model_dict = {}
    for loss_function in loss_function_list:
        for optimizer in optimizer_list:
            for output_activation in output_activation_list:
                model = model_flow_setup(input_shape, num_classes)
                model.add(Activation(output_activation))
                
                model.compile(loss = loss_function,  optimizer = optimizer , metrics = ['accuracy'])
                name = "activation=" + output_activation + "_loss=" + loss_function + "_optimizer=" + optimizer
                print("model generation: " + name)
                print(model.summary())
                model_dict[name] = model
    
    return model_dict




def fine_tune_pretrained_model(input_shape = (224, 224, 3) , num_classes = 2, pre_trained_model_list = ["ResNet"]):
    model_dict = {}
    
    for model_type in pre_trained_model_list:
        if model_type == "ResNet":
            pre_trained_model = ResNet50(weights='imagenet', input_shape = input_shape, include_top = False)
        elif model_type == "VGG19":
            pre_trained_model = VGG19(weights='imagenet', input_shape = input_shape, include_top=False)
        elif model_type == "VGG16":
            pre_trained_model = VGG19(weights='imagenet', input_shape = input_shape, include_top=False)
    
        print("load pre-trainned model weight: " + model_type)
        
        for layer in pre_trained_model.layers:
            layer.trainable = False  # freeze the weight

        
        pre_trained_model.summary()    
        x = pre_trained_model.output       
        x = Flatten(name='flatten')(x)        
        x = Dense(256, activation = "relu", name="fc256") (x)
        #x = Dense(2048, activation = "relu", name="fc2048") (x)
        x = Dense(64, activation = "relu", name="fc64") (x)
        x = Dropout(0.5) (x)    
        x = Dense(num_classes, name="fc_output") (x)
        predictions =  Activation('softmax') (x)
        
        model = Model(input = pre_trained_model.input,  output=predictions)
        model.compile(loss = 'categorical_crossentropy',  optimizer = "rmsprop" , metrics = ['accuracy'])
        print(model.summary())
        model_dict[model_type] = model
    return model_dict



def flow_setup():
    
    one_hot_encoding_dict = label_one_hot_encoding(LABEL_LIST)
    print("label encoding is done.")
    
    
    raw_samples = load_samples(TRAIN_DATA_PATH_LIST, one_hot_encoding_dict)
    print("raw sample loaded.")
    
    #image_shape_counter = raw_data_analysis(raw_samples, one_hot_encoding_dict)
    
    
    #if DO_VISUALIZE:
    #    visualize_image_size_distribution(image_shape_counter)
    
    train_validation_samples, test_samples = train_test_split(raw_samples, test_size = 0.2, random_state = 42)
    train_samples, validation_samples = train_test_split(train_validation_samples, test_size = 0.2, random_state = 42)
    print("train sample count: ", len(train_samples), "\nvalidation sample count: ", len(validation_samples), "\ntest sample count: ", len(test_samples))
    print("sample data example:\n", train_samples[random.randint(0, len(train_samples))])
    
    train_generator = sample_generator(train_samples, batch_size = 100)
    validation_generator = sample_generator(validation_samples, batch_size = 100)
    test_generator = sample_generator(test_samples, batch_size = 100)

    #for i in range(10):
    #    print(next(train_generator))
    #    print(next(test_generator))

    '''
    model = model_setup(input_shape = (200, 200, 3), num_classes = len(LABEL_LIST))
    print("start model fitting...")
    history_object = model.fit_generator(train_generator, samples_per_epoch= int(len(train_samples)), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10, verbose=1)
    score = model.evaluate_generator(test_generator, 1500, max_q_size=10, nb_worker=1, pickle_safe=False)
    model.save("dog_vs_cat_model.h5")
    print(score)
    '''
    if RUN_HOMEBREW_MODEL:    
        print("build home brew model...")
        model_dict =  create_models(input_shape = (224, 224, 3), num_classes = 2, output_activation_list = ['sigmoid', 'softmax'] , loss_function_list = ['categorical_crossentropy' ], optimizer_list = ["rmsprop", "adam"], metrics_list=['accuracy'])
    else:
        print("build model from pre-trainned model...")
        model_dict = fine_tune_pretrained_model(num_classes = 2, pre_trained_model_list = ["VGG16","VGG19","ResNet"])

    print("model build finished.")    
    for name in model_dict.keys():
        model = model_dict[name]
        
        print(name + " fitting started...")
        history_object = model.fit_generator(train_generator, samples_per_epoch= int(len(train_samples)), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
        score = model.evaluate_generator(test_generator, 1500, max_q_size=10, nb_worker=1, pickle_safe=False)
        print(name + " score:", score)
        model.save("./models/" + name + ".h5" )
        print("model " + name + "saved.")

    

        


if __name__ == "__main__":

    # general setting    
    DO_VISUALIZE = False
    TRAIN_DATA_PATH_LIST = ["./train"]
    TEST_DATA_PATH_LIST = ["./test"]
    
    IMAGE_INPUT_SHAPE = (224, 224, 3)
    
    LABEL_LIST = sorted(["dog", "cat"])
    
    PRE_TRAINED_MODEL = ["ResNet"]
    
    RUN_HOMEBREW_MODEL = False
    
    flow_setup()
    
    
