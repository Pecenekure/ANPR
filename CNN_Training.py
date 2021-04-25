import numpy as np
import cv2 as cv
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle




path = 'TrainingData'
testRatio = 0.15
validationRatio = 0.15
imageDimensions = (50,25,3)

images = []
characters = []
myList = os.listdir(path)
print(len(myList))
numberOfClasses = len(myList)

batchSize = 50
epochs = 10
stepsPerEpoch = 1000


######################################načtení obrázků a jejich tagů
for x in range(0,numberOfClasses):
    PicList = os.listdir(path+"/"+ myList[x])
    for y in PicList:
        currentImg = cv.imread(path+"/"+ myList[x] +"/"+y)
        #cv.imshow('erg', currentImg)
        currentImg = cv.resize(currentImg,(imageDimensions[1],imageDimensions[0]))
        #cv.imshow('erg', currentImg)
        images.append(currentImg)
        characters.append(x)
    print(myList[x], end = " ")
print("počet nahraných obrázků:", len(images))

images = np.array(images)
characters = np.array(characters)

print(images.shape)
print(characters.shape)

################################################# Spliting the data
X_train,X_test,y_train,y_test = train_test_split(images, characters, test_size=testRatio)
X_train,X_validation, y_train,y_validation = train_test_split(X_train,y_train,test_size=validationRatio)

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numberOfSamples = []
for x in range(0,numberOfClasses):
    #print("znak {}:".format(x), len(np.where(y_train==x)[0]))
    numberOfSamples.append(len(np.where(y_train==x)[0]))
print(numberOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,numberOfClasses),numberOfSamples)
plt.title('Number of images in each class')
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()

def preprocess(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(preprocess, X_train)))
X_test = np.array(list(map(preprocess, X_test)))
X_validation = np.array(list(map(preprocess, X_validation)))

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0,
                             height_shift_range=0,
                             zoom_range=0.1,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)
y_validation = to_categorical(y_validation, numberOfClasses)

def myModel():
    noOfFilters = 30
    sizeOfFilters1 = (5,5)
    sizeOfFilters2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 300

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilters1, input_shape=(imageDimensions[0], imageDimensions[1], 1), padding='same', activation='relu')))
    #model.add((Conv2D(noOfFilters, sizeOfFilters1, padding='same', activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool, padding='same'))
    model.add((Conv2D(noOfFilters//2, sizeOfFilters2, padding='same', activation='relu')))
    #model.add((Conv2D(noOfFilters//2, sizeOfFilters2, padding='same', activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool, padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSize),
                                           steps_per_epoch=stepsPerEpoch,
                                           epochs=epochs,
                                           validation_data=(X_validation, y_validation),
                                           shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('epochs')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title('Accuracy')
plt.xlabel('epochs')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss= ", score[0])
print("Test Accuracy = ", score[1])

with open ("trained_model_2konvoluce.plk","wb") as modelPickle:
    pickle.dump(model, modelPickle)
    modelPickle.close()

