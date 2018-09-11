import numpy as np
import keras
import math
from keras.optimizers import SGD
import maxflow
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

def train_unary_model(images, gt):
    model = vgg10()
    X_train =  np.zeros((67628, 13, 13, 3))
    Y_train = []
    for i in range(0, 1):
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
        qt=0;
        for j in range(0, width-1):
            for k in range(0, height-1):

                #print 'dd' , i , 'j', j, 'k',k
                Y_train.append(( 1 if gt[i][k,j]==1  else 0 , 0 if gt[i][k,j]==1  else 1))
                X_train[qt]==im_bg[j:j+13, k:k+13]
                qt+=1;

    model.fit(np.array(X_train), np.array(Y_train),                # Train the model using the training set...
          batch_size=200,
          verbose=1, validation_split=0.1)
    model.save("saved_weights.h5")

    return model

def segmentation(unary_model, images):
    for i in range(0, 1):
        im = images[i]
        height, width, channels = im.shape

        
    return [np.zeros(img.shape[:2]) for img in images]
def get_class(tuple):
    if(tuple(0)>tuple(1)):
        return 1
    return 0
def segment_single(im, model):
    
    height, width, channels = im.shape
    im_bg = np.zeros((height+12, width+12, channels))
    im_bg = (im_bg + 1) * 255  

    # Your work: Compute where it should be
    pad_left = 6
    pad_top = 6

    im_bg[pad_top:pad_top + height,
        pad_left:pad_left + width,
        :] = im
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((width, height))
    

    for j in range(1, width-2):
        for k in range(1, height-2):
            profCenter = model.predict( im_bg[j:j+13, k:k+13])
            profL = model.predict( im_bg[j:j+13+1, k:k+13])
            profR = model.predict( im_bg[j:j+13-1, k:k+13])
            profU = model.predict( im_bg[j:j+13, k:k+13+1])
            profD = model.predict( im_bg[j:j+13, k:k+13-1])
            pot_up =get_binary_potential(im_bg[j,k], im_bg[j,k+1]\
            ,get_class(profCenter), get_class(profU) )
            pot_down =get_binary_potential(im_bg[j,k], im_bg[j,k-1]\
            ,get_class(profCenter), get_class(profD) )

            pot_l =get_binary_potential(im_bg[j,k], im_bg[j-1,k]\
            ,get_class(profCenter), get_class(profL) )
            pot_r =get_binary_potential(im_bg[j,k], im_bg[j+1,k]\
            ,get_class(profCenter), get_class(profR) )
            structure = np.array([[0, pot_up, 3],
                        [pot_l, 0, pot_r],
                        [0, pot_down, 0]])
            nodeids = nodes[j-1:j+1, k-1:k+1]
            g.add_grid_edges(nodeids, weights=1, structure=structure, symmetric=False)
            g.add_tende(nodes[j,k], get_unary_potential(profCenter) if profCenter(0)>profCenter(1) else 0,
            get_unary_potential(profCenter) if profCenter(0)<profCenter(1) else 0)
            
            
    g.maxflow()
    graph = g.get_grid_segments(nodes)

            

def vgg10(weights_paty=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=( 13,13,3)))
    model.add(Convolution2D(5, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2))) 
    model.add(Flatten(name='flatten'))
    model.add(Dense(4024, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4024, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax', name='dense_3'))

    model.compile(
        loss='categorical_crossentropy', metrics=['accuracy'],
        optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True))

   
    return model

def get_unary_potential(probability_tuple):
    return -math.log(probability_tuple(0) if probability_tuple(0)>probability_tuple(1)  else probability_tuple(1))

def get_binary_potential(yi, yj,classI, classJ):
    A=0.5
    B=0.5
    sigma = 0.01
    delta = 1 if classI==classJ else 0
    psy = A+B*math.exp(-(np.linalg.dot(yi,yj)**2)/(2*sigma**2))
    return (1-delta)*psy

