import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras import layers, models 

#VGG architecture
def vgg():
    model = Sequential()

    #block 1
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
    model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    #block 2
    model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    #block 3
    model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same', input_shape = (28, 28, 1)))
    model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(256, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides = (2,2)))

    #block 4
    model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    #block 5
    model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(512, (3,3), activation = 'relu', padding = 'same'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    #flattening
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(Dense(10, activation = 'softmax')) # output layer, 10 classes

    return model 
    
vgg_model = vgg()
vgg_model.summary()

#load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocess-normalize
x_train = x_train/255.0
x_test = x_test/255.0

#convert label to categorical 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# print("Shape of x_train:", x_train.shape)
# print("Shape of y_train:", y_train.shape)
# print("Shape of x_test:", x_test.shape)
# print("Shape of y_test:", y_test.shape)

#compile and train model
vgg_model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = 1e-3), metrics = ['accuracy'])
vgg_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs =10, batch_size = 200, verbose = True)

#evaluate model on test data
y_pred = vgg_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

#confusion matrix, preicision, recall, f1 score
confusion = confusion_matrix(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average = 'weighted')
recall = recall_score(y_true, y_pred_classes, average = 'weighted')
f1 = f1_score(y_true, y_pred_classes, average = 'weighted')

print("Confusion Matrix:\n", confusion)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)