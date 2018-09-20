import numpy as np
import math
from keras.optimizers import SGD
import maxflow
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


def train_unary_model(images, gt):
    model = vgg10()
    x_train = []
    y_train = []
    qt = 0
    for i in range(0, 62):
        im = images[i]
        height, width, channels = im.shape
        im_bg = np.zeros((height+12, width+12, channels))
        im_bg = (im_bg + 1) * 255  # e.g., make it white

        # Your work: Compute where it should be
        pad_left = 6
        pad_top = 6

        im_bg[pad_top:pad_top + height,
              pad_left:pad_left + width, :] = im

        for j in range(0, height-1, 5):
            for k in range(0, width-1,5):
                y_train.append((1 if gt[i][j, k] == 1 else 0, 0 if gt[i][j, k] == 1 else 1))
                x_train.append( im_bg[j:j+13, k:k+13])
                qt += 1

    model.fit(np.array(x_train), y_train, batch_size=200,
              nb_epoch=1, verbose=1, validation_split=0.1)

    return model


def segmentation(unary_model, images):
    for i in range(0, 1):
        im = images[i]
        return [segment_single(im, unary_model)]


def get_class(in_tuple):
    if in_tuple[0][0] > in_tuple[0][1]:
        return 1
    return 0


def segment_single(im, model):
    height, width, channels = im.shape
    im_bg = np.zeros((height+12, width+12, channels))
    im_bg = (im_bg + 1) * 255  

    # Your work: Compute where it should be
    pad_left = 6  # type: int
    pad_top = 6

    im_bg[pad_top:pad_top + height,
          pad_left:pad_left + width, :] = im
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((int((height-13)/13), int((width-13)/13)))

    for j in range(27, height-40, 13):
        for k in range(27, width-40, 13):
            prof_center = model.predict(np.reshape(im_bg[j:j+13, k:k+13], (1,  13, 13, 3)))
            print prof_center
            prof_l = model.predict(np.reshape(im_bg[j+1:j+13+1, k:k+13], (1,  13, 13, 3)))
            prof_r = model.predict(np.reshape(im_bg[j-1:j+13-1, k:k+13], (1,  13, 13, 3)))
            prof_u = model.predict(np.reshape(im_bg[j:j+13, k+1:k+13+1], (1,  13, 13, 3)))
            prof_d = model.predict(np.reshape(im_bg[j:j+13, k-1:k+13-1], (1,  13, 13, 3)))
            pot_up = get_binary_potential(im_bg[j, k], im_bg[j, k+1],
                                          get_class(prof_center), get_class(prof_u))
            pot_down = get_binary_potential(im_bg[j, k], im_bg[j, k-1],
                                            get_class(prof_center), get_class(prof_d))

            pot_l = get_binary_potential(im_bg[j, k], im_bg[j-1, k],
                                         get_class(prof_center), get_class(prof_l))
            pot_r = get_binary_potential(im_bg[j, k], im_bg[j+1, k],
                                         get_class(prof_center), get_class(prof_r))
            weights = np.array([[0, pot_up, 0],
                                  [pot_l, 0, pot_r],
                                  [0, pot_down, 0]])
            structure = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
            jg = int(j/13)
            kg = int(k/13)
            node_ids = nodes[jg-1:jg+2, kg-1:kg+2]
            g.add_grid_edges(node_ids, weights=weights, structure=structure, symmetric=False)
            g.add_tedge(nodes[jg, kg], get_unary_potential(prof_center) if prof_center[0][0] < prof_center[0][1] else 0,
                        get_unary_potential(prof_center) if prof_center[0][0] > prof_center[0][1] else 0)
            print 'j=', j, 'k=', k

    g.maxflow()
    ret = np.kron(g.get_grid_segments(nodes), np.ones((13, 13)));
    im_bg = np.zeros((height, width ))


    im_bg[18:18 +213,
    21:21 + 299] = ret
    return  im_bg


def vgg10():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(13, 13, 3)))
    model.add(Convolution2D(5, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4024, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4024, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax', name='dense_3'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True))
    return model


def get_unary_potential(probability_tuple):
    return -math.log(probability_tuple[0][0]
                     if probability_tuple[0][0] > probability_tuple[0][1] else probability_tuple[0][1])


def get_binary_potential(yi, yj, class_i, class_j):
    a = 0.5
    b = 0.5
    sigma = 0.01
    delta = 1 if class_i == class_j else 0
    psy = a+b*math.exp(-(np.dot(yi, yj)**2)/(2*sigma**2))
    return (1-delta)*psy

