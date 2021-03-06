import numpy as np
import random as random
from math import exp as e
from sklearn.metrics import confusion_matrix
from pprint import pprint as pprint

# define number of nodes per layer
input_size = 9
hidden_size = 7
output_size = 6
# learning rate
c = 0.1
# momentum coefficient
alpha = 0.01
# regularization coefficient
lamb = 0.001

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
			for i in range(len(row) - 1):
				line += '{:10.8f},'.format(row[i])
			# end each line with newline
			line += '{:1.0f}\n'.format(row[-1])
			# write line
			file.write(line)

# computes y = f(a) of node from weights and inputs
def node_output (inputs, weights):
	# weight[0] is bias weight
	activation = weights[0]
	for i in range(len(inputs)):
		activation += inputs[i] * weights[i + 1]
	try:
		# compute the simoid(sigmoid?) of the activation
		output = 1 / (1 + e(-1 * activation))
	except OverflowError:
		# catch cases where e^-a overflows
		if activation < 0:
			output = 0
		else:
			output = 1
	return output

# compute error function e with a tolerance
def node_error (d, y):
	# simple error function
	e = d - y
	# maybe?
	tolerance = 0.1
	# if y is within the tolerance of d
	if d == 1 and y >= (d - tolerance):
		e = 0
	# if y is within the tolerance of d
	elif d == 0 and y <= tolerance:
		e = 0
	return e

# compute error for a network ouput
def mse (dj, yj):
	ej = np.array([ node_error(dj[j], yj[j]) for j in range(len(yj)) ])
	return np.sum(ej**2)

# decode target value
def decode_d (d):
	ret = [0, 0, 0, 0, 0, 0]
	if d < 5:
		ret[d - 1] = 1
	else:
		ret[d - 2] = 1
	return ret

def encode_yj (yj):
	ret = 0
	max_index = yj.index(max(yj))
	if max_index + 1 < 4:
		ret = max_index + 1
	else:
		ret = max_index + 2
	return ret

# generate 2D array of random floats
def random_array (shape, lower, upper):
	ret = []
	for j in range(shape[0]):
		ret.append([random.uniform(lower, upper) for i in range(shape[1])])
	return np.array(ret)

# delta term for modifying weights between layers 2 and 1
def delta_oj (d, y):
	return -1 * node_error(d, y) * y * (1 - y)

# delta term for modifying weights between layers 1 and 0
def delta_hi (dj, yj, wji, y):
	ret = 0
	for x in range(len(dj)):
		ret += delta_oj(dj[x], yj[x]) * wji[x]
	return ret * y * (1 - y)

# calculate new weights between layers 2 and 1
def new_delta_wji (weights_ji, old_delta_wji, yj, dj, yi):
	delta_wji = []
	# L2 regularization terms for each weight
	regularize = np.abs(lamb * weights_ji)
	# regularize = lamb * (weights_ji**2)
	# compute momentum values
	momentum = np.zeros((output_size, hidden_size + 1)) if len(old_delta_wji) == 0 else alpha * old_delta_wji
	# add value for bias weight to beginning of yi
	yi_temp = [1]
	yi_temp.extend(yi)
	# for each output node
	for j in range(len(yj)):
		if node_error(dj[j], yj[j]) > 0:
			# delta_oj is a scalar constant for all weights going to node j
			doj = delta_oj(dj[j], yj[j])
			# for each weight ending at node j
			delta_wj = [ (c * doj * yi_temp[i]) + momentum[j][i] + regularize[j][i] for i in range(len(yi_temp)) ]
		else:
			delta_wj = [ 0 for x in range(len(yi_temp)) ]
		# append to entire list
		delta_wji.append(delta_wj)
	return np.array(delta_wji)

# calculate new weights between layes 1 and 0
def new_delta_wih (weights_ji, weights_ih, old_delta_wih, yj, dj, yi, xh):
	delta_wih = []
	# # L2 regularization terms for each weight
	regularize = np.abs(lamb * weights_ih)
	# regularize = lamb * (weights_ih**2)
	# compute momentum values
	momentum = np.zeros((hidden_size, input_size + 1)) if len(old_delta_wih) == 0 else alpha * old_delta_wih
	# add value for bias weight to beginning of xh
	xh_temp = [1]
	xh_temp.extend(xh)
	# for each hidden node
	for i in range(len(yi)):
		# delta_hi is a scalar constant for all weights going to node i
		dhi = delta_hi(dj, yj, weights_ji[:, i], yi[i])
		# for each weight ending at node i
		delta_wi = [ (c * dhi * xh_temp[h]) + momentum[i][h] + regularize[i][h] for h in range(len(xh_temp)) ]
		# append to entire list
		delta_wih.append(delta_wi)
	return np.array(delta_wih)

def train_network ():
	train_data = load_data('training_glass.csv')
	test_data = load_data('testing_glass.csv')

	# initialize weights as random floats, including bias weights
	weights_ih = random_array((hidden_size, input_size + 1), -1 , 1)
	weights_ji = random_array((output_size, hidden_size + 1), -1 , 1)
	# save initial weights to file
	write_2d_array(weights_ih, 'initial_ih_weights.csv')
	write_2d_array(weights_ji, 'initial_ji_weights.csv')

	old_delta_wji = []
	delta_wji = []

	old_delta_wih = []
	delta_wih = []

	training_mse = []
	testing_mse = []
	error_count = []

	for epoch in range(500):
		training_mse.append(0)
		testing_mse.append(0)
		error_count.append(0)
		
		for row in train_data:
			# separate input pattern from ID and target value
			xh = row[1: -1]
			# decode target value
			dj = decode_d(int(row[-1]))
			# layer 1 outputs
			yi = [ node_output(xh, w) for w in weights_ih ]
			# layer 2 outputs
			yj = [ node_output(yi, w) for w in weights_ji ]

			# update MSE for current epoch
			training_mse[epoch] += mse(dj, yj)

			# compute changes in weights
			delta_wji = new_delta_wji(weights_ji, old_delta_wji, yj, dj, yi)
			delta_wih = new_delta_wih(weights_ji, weights_ih, old_delta_wih, yj, dj, yi, xh)

			# save new weights
			weights_ih = weights_ih - delta_wih
			weights_ji = weights_ji - delta_wji
			# save weight changes
			old_delta_wih = delta_wih
			old_delta_wji = delta_wji

		for row in test_data:
			# separate input pattern from ID and target value
			xh = row[1: -1]
			# decode target value
			dj = decode_d(int(row[-1]))
			# layer 1 outputs
			yi = [ node_output(xh, w) for w in weights_ih ]
			# layer 2 outputs
			yj = [ node_output(yi, w) for w in weights_ji ]

			if encode_yj(yj) != int(row[-1]):
				error_count[epoch] += 1

			# update SSE for current epoch
			testing_mse[epoch] += mse(dj, yj)

		print 'Training:' + '{:5.4f} '.format(training_mse[epoch]/len(train_data)) + ' Testing:' + '{:5.4f} '.format(testing_mse[epoch]/len(test_data)) + 'Error Count:' + str(error_count[epoch])
	print 'Training:' + '{:5.4f} '.format(training_mse[0]/len(train_data)) + ' Testing:' + '{:5.4f} '.format(testing_mse[0]/len(test_data)) + 'Error Count:' + str(error_count[0])

	pprint(weights_ji)
	pprint(weights_ih)

# # load and normalize raw data, split into training and testing sets
# data = load_data('GlassData.csv')
# norm_data = normalize_data(data)
# data_sets = split_data(norm_data)
# # write training and testing sets to files
# write_2d_array(data_sets['training'], 'training_glass.csv')
# write_2d_array(data_sets['testing'], 'testing_glass.csv')

train_network()
