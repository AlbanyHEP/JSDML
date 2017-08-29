#	gauss_example.py																		Nicholas Carrara 8/6/17
'''

'''
import JSDML as jsdml
import numpy as np
import random
from matplotlib import pyplot as plt

#	We make a set of 5D gaussians using numpy
signal_mu = 1.0
background_mu = -1.0
sigma = 1.0
num_samples = 100000

signal = [[np.random.normal(signal_mu,sigma,1)[0],
		   np.random.normal(signal_mu,sigma,1)[0],
		   np.random.normal(signal_mu,sigma,1)[0],
		   np.random.normal(signal_mu,sigma,1)[0],
		   np.random.normal(signal_mu,sigma,1)[0]] for i in range(num_samples)]

background = [[np.random.normal(background_mu,sigma,1)[0],
		       np.random.normal(background_mu,sigma,1)[0],
		       np.random.normal(background_mu,sigma,1)[0],
		       np.random.normal(background_mu,sigma,1)[0],
		       np.random.normal(background_mu,sigma,1)[0]] for i in range(num_samples)]

#	Now we'll make some redundant variables
signal_6 = [[np.exp(signal[i][1] + signal[i][2])] for i in range(num_samples)]
signal_7 = [[signal[i][1] + signal[i][2]] for i in range(num_samples)]
background_6 = [[np.exp(background[i][1] + background[i][2])] for i in range(num_samples)]
background_7 = [[background[i][1] + background[i][2]] for i in range(num_samples)]
#	And then add them to the lists
signal = np.concatenate((signal,signal_6,signal_7),axis=1).tolist()
background = np.concatenate((background,background_6,background_7),axis=1).tolist()

data = []
answer = []
for i in range(len(signal)):
	data.append(signal[i])
	answer.append(1.0)
	data.append(background[i])
	answer.append(-1.0)

#	Now let's calculate the JSD for the set of discriminating variables one at a time and train a neural network on them
#	then, we'll calculate the JSD after the network has trained to see if it is optimized.  Then we'll plot the results
num_vars = []
sample_size = int(num_samples/10)
Total_JSD_list = list()
for t in range(len(data[0])):
	JSD_list = list()
	starting_val = 0
	ending_val = sample_size - 1
	#	Here we pick the variables one at a time in succession
	num_vars.append(t)
	temp_data = [[data[j][l] for l in num_vars] for j in range(len(data))]
	for j in range(0,10):
	#	Here we break up the data into 10 subsets to get an average JSD before and after
		temp_scores = list()
		layer_vector = [t+1, 13, 6, 1]
		network = jsdml.nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
		temp_train_x = [temp_data[i] for i in range(starting_val,ending_val)]
		temp_train_y = [answer[i] for i in range(starting_val,ending_val)]
		#	Now separate the training and testing by a 70 / 30 random split
		testing_ratio = int( .3 * sample_size )
		new_train_x = list( temp_train_x[:-testing_ratio] )
		new_train_y = list( temp_train_y[:-testing_ratio] )
		new_test_x = list( temp_train_x[-testing_ratio:] )
		new_test_y = list( temp_train_y[-testing_ratio:] )
		#	Now train the network on the training set
		network.train_network( new_train_x, new_train_y, num_epochs=25, batch=100 )
		#	Then, evaluate the network on the testing results
		results = network.evaluate_network( new_test_x, new_test_y )
		#	We must break up the testing results into separate signal/background so that we can calculate the mutual information
		signal, background = network.split_binary_results( results )	
		temp_train_y = [[temp_train_y[i]] for i in range(len(temp_train_y))]
		#	Now calculating the before and after mutual information
		mutual = jsdml.mi(network.normalize_data(temp_train_x),temp_train_y,k=1)
		new_mutual = network.network_output_jsd(signal, background, neighbors=1)
		#	Now we save all the values we've calculated for this set of sample data
		temp_scores.append(mutual)
		temp_scores.append(new_mutual)
		JSD_list.append(temp_scores)
		print "trial:    ", j
		print "MI before:", mutual
		print "MI after: ", new_mutual
		starting_val += sample_size
		ending_val += sample_size

	#	Now take the mean and standard deviation of the JSD's and AUC's and print them to screen
	before_list = [JSD_list[i][0] for i in range(len(JSD_list))]
	after_list = [JSD_list[i][1] for i in range(len(JSD_list))]
	before_mean = sum(before_list) / 10
	after_mean = sum(after_list) / 10

	before_std = np.std(np.array(before_list))
	after_std = np.std(np.array(after_list))

	print "MI before mean:     ",before_mean
	print "MI before std:      ",before_std
	print "MI after mean:      ",after_mean
	print "MI after std:       ",after_std
	Total_JSD_list.append([before_mean,before_std,after_mean,after_std])


Num_of_vars = [1,2,3,4,5,6,7]
Variable_means = [Total_JSD_list[i][0] for i in range(len(Total_JSD_list))]
Variable_errors = [Total_JSD_list[i][1] for i in range(len(Total_JSD_list))]
Network_means = [Total_JSD_list[i][2] for i in range(len(Total_JSD_list))]
Network_errors = [Total_JSD_list[i][3] for i in range(len(Total_JSD_list))]

plt.figure()
plt.plot(Num_of_vars, Variable_means, color='k', label='jsd_before k=1', linestyle='--')
plt.plot(Num_of_vars, Network_means, color='k', label='jsd_after k=1')
plt.errorbar(Num_of_vars, Variable_means, yerr=Variable_errors,fmt='',color='k',linestyle='--',capsize=2)
plt.errorbar(Num_of_vars, Network_means, yerr=Network_errors,fmt='',color='k',capsize=2)
plt.xticks(np.arange(0, 8, 1.0))
plt.grid()
plt.xlabel('Number of Variables')
plt.ylabel('JSD')
plt.legend( loc='lower right' )
plt.title('5D Gaussian + red. var. JSD vs. Num. of Var.')
plt.show()

