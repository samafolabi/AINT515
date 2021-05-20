import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils

import os
from PIL import Image
from keras.models import load_model

# random seed
np.random.seed(9)

# model settings
nb_epoch = 100
num_classes = 10
batch_size = 20

# the classifications
type_list = ["airplane", "car", "bird", "cat", "deer", "dog", "fox", "house", "boat", "truck"]

# save path for model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'nn_cw_model.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

# load the CIFAR-10 dataset
(trainData, trainLabel), (testData, testLabel) = cifar10.load_data()

# one-hot encoding
trainLabel = np_utils.to_categorical(trainLabel, num_classes)
testLabel = np_utils.to_categorical(testLabel, num_classes)

# neural network
model = Sequential()

# first set of layers
model.add(Conv2D(192,kernel_size=(5,5),activation='relu',
                 strides=1,input_shape=trainData.shape[1:]))
model.add(ZeroPadding2D(2))
model.add(Dense(160, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Dropout(0.5))

# second set of layers
model.add(Conv2D(192, kernel_size=(5,5),activation='relu', strides=1))
model.add(ZeroPadding2D(padding=2))
model.add(Dense(192, activation='relu'))
model.add(Dense(192, activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Dropout(0.5))

# third set of layers
model.add(Conv2D(192, kernel_size=(3,3),activation='relu', strides=1))
model.add(ZeroPadding2D(1))
model.add(Dense(192, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(GlobalAveragePooling2D())

# softmax activation function
model.add(Dense(num_classes, activation='softmax'))

model.summary()


# optimizer for model
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-5)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# format and normalise data
trainData = trainData.astype('float32')
testData = testData.astype('float32')
trainData /= 255
testData /= 255

# build the model and deliver loss and accuracy
history = model.fit(trainData,trainLabel,
                    validation_data=(testData, testLabel),
                    batch_size=batch_size,epochs=nb_epoch, verbose=1)
score = model.evaluate(testData, testLabel, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot accuracy graph
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# plot loss graph
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# Save model
model.save(model_path)
print('Saved trained model at %s ' % model_path)



# load model and test images
# the previous code can be commented out after it has run once
model = load_model(model_path) 

for i in range(1,16):
  pic_name = 'Images for NN/'+str(i)+'.jpg'
  im = Image.open(pic_name)
  im = np.array(im).reshape(1, 32, 32, 3)
  
  result = model.predict(im)
  may_be = np.argmax(result)
  print(type_list[may_be])
  print('\n')
