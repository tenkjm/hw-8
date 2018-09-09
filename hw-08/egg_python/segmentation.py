import numpy as np
import keras
from keras.optimizers import SGD
#import maxflow
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

def train_unary_model(images, gt):
    model = vgg10()
    X_train = []
    Y_train = []
    for i in range(0, 3):
        im = images[i]
        height, width, channels = im.shape

        #print im.shape
        #print gt
        # make canvas
        im_bg = np.zeros((height+12, width+12, channels))
        im_bg = (im_bg + 1) * 255  # e.g., make it white

        # Your work: Compute where it should be
        pad_left = 6
        pad_top = 6

        im_bg[pad_top:pad_top + height,
            pad_left:pad_left + width,
            :] = im
        for j in range(0, width-1):
            for k in range(0, height-1):
                #print 'dd' , i , 'j', j, 'k',k
                Y_train.append(gt[i][k,j])
                X_train.append(im_bg[j:j+13, k:k+13])

    model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=200, epochs=1,
          verbose=1, validation_split=0.1)
    model.save("saved_weights.h5")

    return {}

def segmentation(unary_model, images):
    return [np.zeros(img.shape[:2]) for img in images]

def vgg10(weights_paty=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(3, 13,13)))
    model.add(Convolution2D(5, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2))) 
    model.add(Flatten(name='flatten'))
    model.add(Dense(196, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(196, activation='relu', name='dense_2'))
    
    model.add(Dense(2, activation='softmax', name='dense_3'))

    model.compile(
        loss='categorical_crossentropy', metrics=['accuracy'],
        optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True))

   
    return model
