# import modules
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.keras.datasets import reuters # the dataset
from tensorflow.keras.models import Sequential # assuming this means
# a simple fully connected MLP? 
# should look into what other models there are
from tensorflow.keras.layers import Dense, Dropout, Activation
# libraries used to define the model 
	# dense: assuming it contains functions for defining a hidden layer
	# dropout: functions for dropout regularization
	# activation: functions for activation functions (ReLu, Sigmoid)
from tensorflow.keras.preprocessing.text import Tokenizer
# tokenizing the imported data set
# tokenizing is just breaking down or reformatting the data in a different
# data type



# the x data
# it is stored as a LIST of INTEGER STRINGS
# reuters data set contains many samples
# each sample is a news article
# here we are spliting the available samples into 0.2 and 0.8
# each sample is represented as a string of integers ('1435...')
	# in our case we get the dictionary of the most common 10000 words
	# key: the word, value: integer corresponding to the commonality rank
	# we reference this dictionary to label each word


# the y data
# it is stored as a LIST of INTEGERS
# these are the classes 
# in this case 47 because there are 47 different topics that
# a news article could be
# each element in the list corresponds to a sample/news article
# each element is an integer from 1-47 corresponding to the class of the
# article

# load reuters dataset

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train) + 1 # number of classes 1 more than the 
	# largest # class (ie. 47) because the class numbers in the list is 
	# 0 indexed
print(num_classes, 'classes')	

# vectorize sequence Data and One hot encode class labels

tokenizer = Tokenizer(num_words=10000) # this tokenizer object will be
# able to tokenize all samples if it only contains top 10000 most common
# words

# the x datasets will be converted to a matrix with
# row= # of samples/news articles
# col= 10000
	# each element in the column represents a top 10000 word
	# binary, so 1 if the article contains the word, 0 if not

x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# y data set converted to matrix as well
# row = # samples 
# col = 47
	# there are 47 classes
	# 1 if it is that class, 0 if not
print('Convert class vector to binary class matrix for use with categorical_crossentropy')
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# Build MLP Model

# Sequential is another name for feedforward models (only goes one direction)
model = Sequential()
# A Dense layer = fully connected 
model.add(Dense(512, input_shape=(10000,))) # first hidden layer
# input is 10000 because each of the 512 perceptron/node will process
# one sample at a time
# and one x sample can be represented by a single 10000 element long vector
model.add(Activation('relu')) # activation function aftter layer 1
model.add(Dropout(0.5)) # adding regularization 
model.add(Dense(num_classes)) # output layer should be same number as
# number of classes (each node in this layer represents an element in the
# y vector)
model.add(Activation('softmax')) # changes final layer so that all nodes
# add up to 1

model.summary() # show summary of model


# train model

# import additional module for adding early stopping when training model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# instantiating an early stopping callback before we train the model
# patience = 3, so the model will wait 3 training epochs before stopping
# training
# mode = min, because we are trying to minimize loss
early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=1, mode='min')
# creating list to store all the callbacks the model should have
# in our case we only added the earlystopping callback
callbacks = [early_stopping]

# tthe optimizer is the training algorithm model will use
	# could've selected stochastic gradient descent here
	# but adam is a more advanced algorithm that is better in many cases
# categorical crossentropy is just the way to measure loss in we are
# dealing with categorizing into more than 2 groups
# accuracy has similar formula to precision, recall, and such
	# this metric is used when testing on test set and assesing performance
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit initiates trainin
# this history variable will keep record of the loss when training
# accuracy on test set, and loss on validation
# batch size is 32, so instead of averaging loss of all samples before it
# takes a step, it only uses 32 samples
# we also specifiy that we will use 0.1 of our training data for validation
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=callbacks)

# evaluate model on test data

score = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
print('Test loss: {:4f}\nTest Accuracy: {:4f}'.format(score[0], score[1]))

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
