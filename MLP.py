# one hot encoding data for MLP model 

import sys
import gzip
import mcb185
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # assuming this means
from tensorflow.keras.layers import Dense, Dropout, Activation
from Bio import SeqIO

fasta = sys.argv[1]
gff = sys.argv[2]
d_fasta = sys.argv[3]
a_fasta = sys.argv[4]
n_d_fasta = sys.argv[5]
n_a_fasta = sys.argv[6]

# input: d_fasta, a_fasta, n_d_fasta, or n_a_fasta
def one_hot_encode_x(fasta, window_len):
	# list of 4x9 numpy arrays
	one_hot_matrix_list= []

	# turn list of seq into list of 4x9 numpy array
	one_hot_dict = {
		'A': np.array([1, 0, 0, 0]),
		'C': np.array([0, 1, 0, 0]),
		'T': np.array([0, 0, 1, 0]),
		'G': np.array([0, 0, 0, 1])
	}
	
	for defline, seq in mcb185.read_fasta(fasta):
		seq = seq.upper()

		# you can cast a list of numpy arrays into a numpy matrix
		matrix = np.array([one_hot_dict[base] for base in seq])
		one_hot_matrix_list.append(matrix)

	return one_hot_matrix_list


	# also should create a numpy array for the y values
	# should either be 0 or 1

def one_hot_encode_y(x_matrix_list, pos_or_neg):
	n = len(x_matrix_list)
	if pos_or_neg == '+':
		y_arr = np.ones(n)
	if pos_or_neg == '-':
		y_arr = np.zeros(n)

	return y_arr


# pos donor x and y 
pos_donor_x_matrix_list = one_hot_encode_x(d_fasta, 9)
pos_donor_y_arr = one_hot_encode_y(pos_donor_x_matrix_list, '+')

# neg donor x and y
neg_donor_x_matrix_list = one_hot_encode_x(n_d_fasta, 9)
neg_donor_y_arr = one_hot_encode_y(neg_donor_x_matrix_list, '-')

# combined x data, cast to numpy array
# flatten 4x9 array into 36x1 array

x_data = np.array(pos_donor_x_matrix_list + neg_donor_x_matrix_list)
x_data = x_data.reshape((x_data.shape[0], -1))
y_data = np.concatenate((pos_donor_y_arr, neg_donor_y_arr))


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Build Model

model = Sequential()
 
model.add(Dense(300, input_shape=(36, ))) # first hidden layer

model.add(Activation('relu'))

model.add(Dropout(0.1))

model.add(Dense(150))

model.add(Activation('relu'))

model.add(Dense(1)) # output layer should have 1 node/perceptron 

model.add(Activation('sigmoid'))

model.summary()



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=14, verbose=1, mode='min')

callbacks = [early_stopping]

# configure model
model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy'])


# train model
history = model.fit(x_train, y_train,
					epochs=100,
					batch_size=10000,
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















