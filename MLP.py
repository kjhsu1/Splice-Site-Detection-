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

def extract_positive_coordinates(fasta, gff, acceptor_base_length, d_lim_chrom, d_lim_num, a_lim_chrom, a_lim_num):
	# initialize donor and acceptor coordinate dictionaries
	donor_coords = {}
	acceptor_coords = {}

	# the log odd scores for all windows are stored as 1-based index
		# NOTE: log odds scores are all on the positive strand AND they are stored in
		# different dictionaries for every chromosome

	# create separate lists for every chromosome/defline
	for defline, seq in mcb185.read_fasta(fasta):
		defline_words = defline.split()
		donor_coords[defline_words[0]] = []
		acceptor_coords[defline_words[0]] = []

	# iterate through gff to store
	with gzip.open(gff, 'rt') as file:
		for line in file:
			words = line.split()
			if words[1] != 'RNASeq_splice': continue
			# extract intron information 
			chrom = words[0]
			start = int(words[3])
			end = int(words[4])
			strand = words[6]

			# find window index for donor and acceptor
			donor_index = start
			acceptor_index = end - acceptor_base_length + 1

			# only collecting + strand for now
			if strand != '+': continue

			# for loop to store in the correct dictionary
			for key in donor_coords.keys():
				if key == chrom:
					donor_coords[key].append(donor_index)
					acceptor_coords[key].append(acceptor_index)

	# TESTING
	'''
	print('before limiting')
	for key, num in zip(lim_chrom, lim_num):
		print(key)
		print(len(donor_coords[key]))
		print(len(acceptor_coords[key]))
		print()
	'''

	# create the chrom limited d and a pos dict if requested
	# donor
	if d_lim_chrom != 'NA':
		limited_pos_donor_coords = {}

		for defline, seq in mcb185.read_fasta(fasta):
			defline_words = defline.split()
			limited_pos_donor_coords[defline_words[0]] = []
		# donor
		for key in donor_coords.keys():
			if key not in d_lim_chrom: continue
			# only take the coords if it is the chrom requested
			limited_pos_donor_coords[key] = donor_coords[key]
			# get rid of repeats
			limited_pos_donor_coords[key] = list(set(limited_pos_donor_coords[key]))

	# acceptor
	if a_lim_chrom != 'NA':
		limited_pos_acceptor_coords = {}

		for defline, seq in mcb185.read_fasta(fasta):
			defline_words = defline.split()
			limited_pos_acceptor_coords[defline_words[0]] = []
		# acceptor 
		for key in acceptor_coords.keys():
			if key not in a_lim_chrom: continue
			# only take the coords if it is the chrom requested
			limited_pos_acceptor_coords[key] = acceptor_coords[key]
			limited_pos_acceptor_coords[key] = list(set(limited_pos_acceptor_coords[key]))


	# if abs number of coord limitation is also requested...
	# donor
	if d_lim_num != 'NA':
		# donor
		for key, num in zip(d_lim_chrom, d_lim_num):
			# first need to get rid of repeats
			limited_pos_donor_coords[key] = list(set(limited_pos_donor_coords[key]))


			# slice appropriately
			words = num.split()
			# when need only one range
			if len(words) == 2:
				start = int(words[0]) - 1 
				end = int(words[1])
				limited_pos_donor_coords[key] = limited_pos_donor_coords[key][start:end]
			
			# when need two ranges 
			if len(words) == 4:
				start1 = int(words[0]) - 1
				end1 = int(words[1])
				start2 = int(words[2]) - 1
				end2 = int(words[3])
				# coord list 1
				coordlist1 = limited_pos_donor_coords[key][start1:end1]
				# coord list 2
				coordlist2 = limited_pos_donor_coords[key][start2:end2]

				# join both lists
				limited_pos_donor_coords[key] = coordlist1 + coordlist2

	# if abs number of coord limitation is also requested...
	# acceptor
	if a_lim_num != 'NA':
		# acceptor
		for key, num in zip(a_lim_chrom, a_lim_num):
			# first need to get rid of repeats
			limited_pos_acceptor_coords[key] = list(set(limited_pos_acceptor_coords[key]))


			# slice appropriately
			words = num.split()
			# when need only one range
			if len(words) == 2:
				start = int(words[0]) - 1 
				end = int(words[1])
				limited_pos_acceptor_coords[key] = limited_pos_acceptor_coords[key][start:end]
			
			# when need two ranges 
			if len(words) == 4:
				start1 = int(words[0]) - 1
				end1 = int(words[1])
				start2 = int(words[2]) - 1
				end2 = int(words[3])
				# coord list 1
				coordlist1 = limited_pos_acceptor_coords[key][start1:end1]
				# coord list 2
				coordlist2 = limited_pos_acceptor_coords[key][start2:end2]

				# join both lists
				limited_pos_acceptor_coords[key] = coordlist1 + coordlist2


	# TESTING
	'''
	for key, num in zip(lim_chrom, lim_num):
		print(key)
		print(len(limited_pos_donor_coords[key]))
		print(len(limited_pos_acceptor_coords[key]))
		print()
	'''

	# return limited versions
	if d_lim_chrom != 'NA' and a_lim_chrom != 'NA':
		return limited_pos_donor_coords, limited_pos_acceptor_coords

	if d_lim_chrom == 'NA' and a_lim_chrom != 'NA':
		return donor_coords, limited_pos_acceptor_coords

	if d_lim_chrom != 'NA' and a_lim_chrom == 'NA':
		return limited_pos_donor_coords, acceptor_coords

	return donor_coords, acceptor_coords

def extract_negative_coords(donor_coordinates, acceptor_coordinates, fasta, donor_base_length, acceptor_base_length, d_lim_chrom, d_lim_num, a_lim_chrom, a_lim_num):
	# initialize donor and acceptor dictionaries of lists 
	negative_donor_coords = {}
	negative_acceptor_coords = {} 

	# create separate lists for every chromosome/defline (ie. I, MtDNA, etc)
	for defline, seq in mcb185.read_fasta(fasta):
		defline_words = defline.split()
		negative_donor_coords[defline_words[0]] = []
		negative_acceptor_coords[defline_words[0]] = []

	# iterate one chromosome at a time 
	for defline, seq in mcb185.read_fasta(fasta):
		defline_words = defline.split()
		chrom = defline_words[0]

		# extract the appropriate donor and acceptor coordinates list for the current
		# chromosome
		# Note: these coordinates are 1-based index 
		current_donor_list = donor_coordinates[chrom]
		current_acceptor_list = acceptor_coordinates[chrom]

		# window through each base in the seq
		# donor window first
		for i in range(len(seq) - donor_base_length +1):
			# fix to 1-based index
			base_number = i + 1

			# if base number matches with any of the donor coordinates, skip
			if base_number in current_donor_list: continue

			# if not in donor list, check if first 2 bases are GT
			# if yes, store as negative coord

			if seq[base_number - 1:base_number + 1].upper() == 'GT':
				negative_donor_coords[chrom].append(base_number)

		# after finished with windowing through the donor for the particular chrom,
		# window through the negative
		for i in range(len(seq) - acceptor_base_length +1):
			if seq[i+acceptor_base_length-2:i+acceptor_base_length] == 'AG':
				if i+1 not in current_acceptor_list:
					negative_acceptor_coords[chrom].append(i+1)

		
	# create the chrom limited d and a negative dict if requested
	# donor
	if d_lim_chrom != 'NA':
		limited_neg_donor_coords = {}

		for defline, seq in mcb185.read_fasta(fasta):
			defline_words = defline.split()
			limited_neg_donor_coords[defline_words[0]] = []
		for key in negative_donor_coords.keys():
			if key not in d_lim_chrom: continue
			# only take the coords if it is the chrom requested
			limited_neg_donor_coords[key] = negative_donor_coords[key]
			limited_neg_donor_coords[key] = list(set(limited_neg_donor_coords[key]))

	# acceptor
	if a_lim_chrom != 'NA':
		limited_neg_acceptor_coords = {}

		for defline, seq in mcb185.read_fasta(fasta):
			defline_words = defline.split()
			limited_neg_acceptor_coords[defline_words[0]] = []
		# acceptor 
		for key in negative_acceptor_coords.keys():
			if key not in a_lim_chrom: continue
			# only take the coords if it is the chrom requested
			limited_neg_acceptor_coords[key] = negative_acceptor_coords[key]
			limited_neg_acceptor_coords[key] = list(set(limited_neg_acceptor_coords[key]))

	# if abs number of coord limitation is also requested...
	# donor
	if d_lim_num != 'NA':
		# donor
		for key, num in zip(d_lim_chrom, d_lim_num):
			# first need to get rid of repeats
			limited_neg_donor_coords[key] = list(set(limited_neg_donor_coords[key]))


			# slice appropriately
			words = num.split()
			# when need only one range
			if len(words) == 2:
				start = int(words[0]) - 1 
				end = int(words[1])
				limited_neg_donor_coords[key] = limited_neg_donor_coords[key][start:end]
			
			# when need two ranges 
			if len(words) == 4:
				start1 = int(words[0]) - 1
				end1 = int(words[1])
				start2 = int(words[2]) - 1
				end2 = int(words[3])
				# coord list 1
				coordlist1 = limited_neg_donor_coords[key][start1:end1]
				# coord list 2
				coordlist2 = limited_neg_donor_coords[key][start2:end2]

				# join both lists
				limited_neg_donor_coords[key] = coordlist1 + coordlist2

	# if abs number of coord limitation is also requested...
	# acceptor
	if a_lim_num != 'NA':
		# acceptor
		for key, num in zip(a_lim_chrom, a_lim_num):
			# first need to get rid of repeats
			limited_neg_acceptor_coords[key] = list(set(limited_neg_acceptor_coords[key]))


			# slice appropriately
			words = num.split()
			# when need only one range
			if len(words) == 2:
				start = int(words[0]) - 1 
				end = int(words[1])
				limited_neg_acceptor_coords[key] = limited_neg_acceptor_coords[key][start:end]
			
			# when need two ranges 
			if len(words) == 4:
				start1 = int(words[0]) - 1
				end1 = int(words[1])
				start2 = int(words[2]) - 1
				end2 = int(words[3])
				# coord list 1
				coordlist1 = limited_neg_acceptor_coords[key][start1:end1]
				# coord list 2
				coordlist2 = limited_neg_acceptor_coords[key][start2:end2]

				# join both lists
				limited_neg_acceptor_coords[key] = coordlist1 + coordlist2


	# return limited versions
	if d_lim_chrom != 'NA' and a_lim_chrom != 'NA':
		return limited_neg_donor_coords, limited_neg_acceptor_coords

	if d_lim_chrom == 'NA' and a_lim_chrom != 'NA':
		return negative_donor_coords, limited_neg_acceptor_coords

	if d_lim_chrom != 'NA' and a_lim_chrom == 'NA':
		return limited_neg_donor_coords, negative_acceptor_coords

	# return un-limited version otherwise
	return negative_donor_coords, negative_acceptor_coords


	# load genome as dictionary with values as SeqIO object
	# replace genome.fasta with actual file path


# FOR REAL GENOME
'''
d_lim_chrom = ['I', 'II', 'III', 'IV', 'V', 'X']
d_lim_num = ['1 256 385 512', '1 254 382 508', '1 120 181 240', '1 194 292 388', '1 270 406 540', '1 174 262 348']
a_lim_chrom = ['I', 'II', 'III', 'IV', 'V', 'X']
a_lim_num = ['1 238 358 476', '1 254 382 508', '1 122 184 244', '1 174 262 348', '1 244 367 488', '1 160 241 320']
'''

d_lim_chrom = ['I', 'II', 'III', 'IV', 'V', 'X']
d_lim_num = 'NA'
a_lim_chrom = ['I', 'II', 'III', 'IV', 'V', 'X']
a_lim_num = 'NA'

donor_coordinates, acceptor_coordinates = extract_positive_coordinates(fasta, gff, 25, d_lim_chrom, d_lim_num, a_lim_chrom, a_lim_num)

d_lim_chrom = ['I', 'II', 'III', 'IV', 'V', 'X']
d_lim_num = ['1 515', '1 509', '1 243', '1 390', '1 540', '1 350']
a_lim_chrom = ['I', 'II', 'III', 'IV', 'V', 'X']
a_lim_num = ['1 478', '1 510', '1 247', '1 349', '1 491', '1 320']

neg_donor_coordinates, neg_acceptor_coordinates =  extract_negative_coords(donor_coordinates, acceptor_coordinates, fasta, 9, 25, d_lim_chrom, d_lim_num, a_lim_chrom, a_lim_num)


# open genome as dictionary of SeqIO sequence objects
with gzip.open('../Data/1pct_elegans.fa.gz', 'rt') as handle:
	    genome_dict = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))# Function to get sequence

# convert dictionary of donor coord list into list of tuples
def to_tuples(coord_dictionary, window_len):
	# create list
	tuple_list = []
	for key in coord_dictionary.keys():
		for coord in coord_dictionary[key]:
			if key == 'I':
				tuple_list.append((key, coord, coord+window_len-1))
			if key == 'II':
				tuple_list.append((key, coord, coord+window_len-1))
			if key == 'III':
				tuple_list.append((key, coord, coord+window_len-1))
			if key == 'IV':
				tuple_list.append((key, coord, coord+window_len-1))
			if key == 'V':
				tuple_list.append((key, coord, coord+window_len-1))
			if key == 'X':
				tuple_list.append((key, coord, coord+window_len-1))		
	return tuple_list

# Function to get sequence
# ASSUMES ALL COORDS ARE ON POS STRAND
# 'chrom' is just the roman numeral 
def get_sequence(chrom, start, end):
	return str(genome_dict[chrom].seq[start-1:end])

# either acceptor or donor
# either pos or neg
# agnostic function
def one_hot_encode_x(coordinates, window_len):
	# list of tuples
	tuples_list = to_tuples(coordinates, window_len)


	# USE LIST OF TUPLES OF COORDINATES TO CONVERT INTO NUMPY ARRAY

	# ASSUMES ALL COORDS ARE ON POS STRAND

	# create list of extracted sequences
	seq_list = [get_sequence(chrom, start, end) for chrom, start, end in tuples_list]

	# turn list of seq into list of 4x9 numpy array
	one_hot_dict = {
		'A': np.array([1, 0, 0, 0]),
		'C': np.array([0, 1, 0, 0]),
		'T': np.array([0, 0, 1, 0]),
		'G': np.array([0, 0, 0, 1])
	}


	# list of 4x9 numpy arrays
	one_hot_matrix_list= []

	# iterate through all seqs
	# for each seq, look at each base
	# if base matches one of the key in dict, store appropriate np.array
	# you can cast a list of numpy arrays into a numpy matrix
	for seq in seq_list:
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
pos_donor_x_matrix_list = one_hot_encode_x(donor_coordinates, 9)
# print(one_hot_pos_donor_x_matrix)
pos_donor_y_arr = one_hot_encode_y(pos_donor_x_matrix_list, '+')

# neg donor x and y
neg_donor_x_matrix_list = one_hot_encode_x(neg_donor_coordinates, 9)
neg_donor_y_arr = one_hot_encode_y(neg_donor_x_matrix_list, '-')

# combined x data, cast to numpy array
# flatten 4x9 array into 36x1 array

x_data = np.array(pos_donor_x_matrix_list + neg_donor_x_matrix_list)
x_data = x_data.reshape((x_data.shape[0], -1))
y_data = np.concatenate((pos_donor_y_arr, neg_donor_y_arr))


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Build Model

model = Sequential()
 
model.add(Dense(512, input_shape=(36, ))) # first hidden layer

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1)) # output layer should have 1 node/perceptron 

model.add(Activation('sigmoid'))

model.summary()



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

callbacks = [early_stopping]

# configure model
model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy'])


# train model
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















