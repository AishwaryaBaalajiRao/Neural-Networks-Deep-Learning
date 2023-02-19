import scipy.io as io
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow
from tensorflow.keras.utils import to_categorical

# Assign batch size, epoch and no of classes
batchSize = 128
total_classes = 10
epochs = 20
lr = 0.01

# Assigning the input image dimensions (as per given dataset)
imgRows, imgCols = 32, 32

# Reading the data using scipy 
trX = io.loadmat('train_32x32.mat')['X']
trY = io.loadmat('train_32x32.mat')['y']
tsX = io.loadmat('test_32x32.mat')['X']
tsY = io.loadmat('test_32x32.mat')['y']

# Normalizing the data to fit in 0-1 range (one-hot vector encoding)
xTrain = trX.astype('float32')
xTest = tsX.astype('float32')
xTrain /= 255
xTest /= 255

# Converting class vectors to binary class matrices
yTrain = to_categorical(trY, total_classes)
yTest = to_categorical(tsY, total_classes)

input_shape = (imgRows, imgCols, 3)

# Building the CNN model following the architecture in the project
cnnModel = Sequential()

cnnModel.add(Conv2D(64, kernel_size = (5, 5),
                 activation = 'relu',
                 padding='same', strides=(1,1),
                 input_shape = input_shape))
cnnModel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnnModel.add(Conv2D(64, kernel_size = (5, 5), padding='same',
                 strides=(1,1), activation = 'relu'))
cnnModel.add(MaxPooling2D(pool_size =(2, 2), strides = (2, 2)))

cnnModel.add(Conv2D(128, kernel_size = (5, 5), padding='same', 
                    strides=(1,1), activation = 'relu'))

cnnModel.add(Flatten())
cnnModel.add(Dense(3072, activation = 'relu'))
cnnModel.add(Dense(2048, activation = 'relu'))

cnnModel.add(Dense(total_classes, activation = 'softmax'))

cnnModel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.SGD(learning_rate=lr),
              metrics=['accuracy'])

Train_cnn = cnnModel.fit(xTrain, yTrain,
          batch_size = batchSize,
          epochs = epochs,
          verbose = 1,
          validation_data = (xTest, yTest))
score = cnnModel.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plotting Test vs Train Loss and Accuracy as a function of epochs using matplotlib 
from matplotlib import pyplot as plt
train = []
test = []

for item in Train_cnn.history['loss']:
  train.append(item)
for item in Train_cnn.history['val_loss']:
  test.append(item)

plt.plot(train)
plt.plot(test)
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train - loss', 'Test - val_loss'], loc='upper left')
plt.show()

train = []
test = []

for item in Train_cnn.history['accuracy']:
  train.append(item)
for item in Train_cnn.history['val_accuracy']:
  test.append(item)

plt.plot(train)
plt.plot(test)
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train - accuracy', 'Test - val_accuracy'], loc='upper left')
plt.show()