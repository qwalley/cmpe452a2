import numpy as np
import random as random
from math import exp as e
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from pprint import pprint as pprint

# load raw data from csv file, strip new lines and format as floats
def load_data (filename):
	data_array = []
	
	with open(filename, "r") as file:
		# remove column headings
		headers = file.readline()
		# for each input pattern
		for line in file:
			# split comma separated data and remove '\n'
			split_line = line[:-1].split(',')
			numeric_list = [float(val) for val in split_line]
			# store numeric data
			data_array.append(numeric_list)
	# return data in numpy array
	return np.array(data_array)

# get z-score of each data point with respect to its column
def normalize_data (np_array):
	# get mean of each column
	mean = np.mean(np_array, axis=0)
	# get standard deviation of each column
	std_dev = np.std(np_array, axis=0)

	def normalize_row(np_row):
		# normalize all but first and last value in a row, first value is ID, last value is d
		row = [(val - mean[i]) / std_dev[i] if (i < (len(np_row) - 1) and i > 0) else val for (i,), val in np.ndenumerate(np_row)]
		return row
	
	return np.apply_along_axis(normalize_row, axis=1, arr=np_array)

# divide data patterns into 70% training, 30% testing with even distribution of each class
def split_data (np_array):
	data_by_class_dict = {}
	data_sets = { 'training': [], 'testing': [] }
	# group input patterns into separate arrays by their class
	for row in np_array:
		# create dictionary key from class number
		class_id = 'class' + str(int(row[-1]))
		if class_id in data_by_class_dict:
			# add pattern to dict under its class
			data_by_class_dict[class_id].append(row)
		else:
			# or create class entry and add the row
			data_by_class_dict[class_id] = [row]
	# for each class list
	for class_id in data_by_class_dict:
		# calculate the 70% index
		split_point = int(len(data_by_class_dict[class_id]) * 0.7)
		# add 0-70% to training data
		data_sets['training'].extend(data_by_class_dict[class_id][:split_point])
		# add 70-100% to testing data
		data_sets['testing'].extend(data_by_class_dict[class_id][split_point:])
	# randomize order of training data
	random.shuffle(data_sets['training'])
	return data_sets

# writes 2D arrays to file as comma separated values
def write_2d_array (array, filename):
	# open file for writing
	with open(filename, 'w') as file:
		# write dummie header line
		file.write('------header-------\n')
		# format each row
		for row in array:
			line = ''
			# separate values by comma
			line += '{:3.0f},'.format(row[0])
			for i in range(1, len(row) - 1):
				line += '{:10.8f},'.format(row[i])
			# end each line with newline
			line += '{:1.0f}\n'.format(row[-1])
			# write line
			file.write(line)

# computes y = f(a) of node from weights and inputs
def node_output (inputs, weights):
	activation = weights[0]
	for i in range(len(inputs)):
		# weight[0] is bias weight
		activation += input[i] * weights[i + 1]
	# compute the simoid(sigmoid?) of the activation
	output = 1 / (1 + e(-1 * activation))
	return output

# compute error function e with a tolerance
def error (d, y):
	e = d - y
	# maybe?
	tolerance = 0.05
	# if y is within the tolerance of d
	if d == 1 and y >= (d - tolerance):
		e = 0
	# if y is within the tolerance of d
	elif d == 0 and y <= tolerance:
		e = 0
	return e

# decode target value
def decode_d (d):
	ret = [0, 0, 0, 0, 0, 0]
	if d < 5:
		ret[d - 1] = 1
	else:
		ret[d - 2] = 1
	return ret

# load and normalize raw data, split into training and testing sets
data = load_data('GlassData.csv')
norm_data = normalize_data(data)
data_sets = split_data(norm_data)
# write training and testing sets to files
write_2d_array(data_sets['training'], 'training_glass.csv')
write_2d_array(data_sets['testing'], 'testing_glass.csv')
