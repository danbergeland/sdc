import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import csv
import json

from cv2 import resize, flip,cvtColor, COLOR_RGB2HSV

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from math import floor

training_file = 'data/driving_log.csv'

imgYdim = 64
imgXdim = 64
    
def processImg(imgPath, flipDis=False):
    #img is 160 x 320
    topVal =50
    bottomVal=140
    im = mpimage.imread(imgPath.strip())
    im = im[topVal:bottomVal,:,:]
    im = cvtColor(resize(im,(imgXdim,imgYdim)), COLOR_RGB2HSV)
    if flipDis:
        im = flip(im, 1)
    #im = randomBrightnessShift(im)
    #im = regularize(im)
    return im


with open(training_file,'r') as f:
    reader = csv.reader(f, delimiter=',') 
    data = list(reader)
    #split 10% training data
    tsplit = int(floor(len(data)*.1))
    np.random.shuffle(data)
    val,train = data[:tsplit],data[tsplit:]
    
def generateFromDriveData(batch=64,valData=False):
    while 1:   
        if not valData: 
            imgs = np.array(np.empty([batch,imgYdim,imgXdim,3]))           
            ys = np.array(np.empty([batch,1]))
            
            for i in range(batch):        
                #select center, left or right
                sideSelect=np.random.randint(0,2)
                steerOffset=0
                if sideSelect == 1:
                    steerOffset=.25
                elif sideSelect ==2:
                    steerOffset=-.25
                y=0
                #randomly allow steer angles greater than 0 to filter out straight data
                angleSelect= np.random.choice([0,.03,.03,.05,.3,.35])
                while abs(y) <= angleSelect:
                    lineNum = np.random.randint(len(train))
                    line = train[lineNum]
                    img1path = line[sideSelect].lstrip()
                    y = float(line[3])+steerOffset

                #flip image and y value on every other image
                if i%2==0:
                    img = processImg(img1path,True)
                    y *= -1
                else:
                    img= processImg(img1path,False)
                #add noise to y values +/- .04
                scaler = .04*(np.random.random()-.5)
                y += scaler
                imgs[i] = img
                ys[i] = np.array([y])
                samples = i
            imgs = np.reshape(imgs,[-1,imgYdim,imgXdim,3])
            ys = np.reshape(ys,[-1,1])
            yield imgs, ys

        else:
            lineNum = np.random.randint(len(val))
            line = val[lineNum]
            img1path = line[0]
            img = processImg(img1path)
            y = float(line[3])
            yield np.array([img]),np.reshape(np.array([y]),[-1,1])

def makeModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(imgYdim, imgXdim,3), output_shape=(imgYdim, imgXdim,3)))
    model.add(Convolution2D(3,1,1,border_mode='same',activation='relu'))
    model.add(Convolution2D(32,3,3,border_mode='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))       
    model.add(Dropout(.6))    
    model.add(Convolution2D(32,3,3,border_mode='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))       
    model.add(Dropout(.6))  
    model.add(Convolution2D(64,3,3,border_mode='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.6))
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(.5))   
    model.add(Dense(1))
    adam = Adam()
    model.compile(loss='mean_squared_error',optimizer=adam)
    return model

if __name__ == '__main__':
    model = makeModel()
    print(model.summary())
    trainSamplesPerEpoch = 10000
    epochs = 2
    batch_size = 256
    jenny = generateFromDriveData(batch_size)
    valJenny = generateFromDriveData(batch=1,valData=True)
    history = model.fit_generator(jenny, samples_per_epoch=trainSamplesPerEpoch, nb_epoch=epochs, validation_data=valJenny, nb_val_samples=1000)

    jModel = model.to_json()
    with open('drivemodel.json', 'w') as outfile:
        json.dump(jModel, outfile)
    model.save_weights('drivemodel.h5')

