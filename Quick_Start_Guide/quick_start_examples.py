#	quick_start_examples.py																		Nicholas Carrara 8/6/17
'''
These are meant to be quick examples of the functionality of the JSDML code in 5-10 lines.
'''
import JSDML as jsdml
import numpy as np
import random
from matplotlib import pyplot as plt

#	Example 1 - 5D Gausians
train_x, train_y, test_x, test_y = jsdml.create_feature_sets_and_labels( "QuickStartGaussSignal.csv",
																	  "QuickStartGaussBackground.csv", 
																	  num_of_vars=5, 
																	  test_size=0.3, 
																	  file_size=0.999, 
																	  var_set=[0,1,2,3,4] )
layer_vector = [5, 11, 4, 1]
network = jsdml.nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
#	Now we train the network on the training data and evaluate the testing data
network.train_network( train_x, train_y, num_epochs=25, batch=100 )
results = network.evaluate_network( test_x, test_y )
#	Now we evaluate the mutual information on the input and output
train_y = [[train_y[i]] for i in range(len(train_y))]
mutual = jsdml.mi(network.normalize_data(train_x),train_y,k=1)
signal, background = network.split_binary_results( results )
new_mutual = network.network_output_jsd(signal, background, neighbors=3)
print "MI before: ", mutual
print "MI after:  ", new_mutual
#	This section generates the ROC curves and calculates the area under curve	
network.plot_network_output(results, symmetric=False)



#	Example 2 - Magic Data
train_x, train_y, test_x, test_y = jsdml.create_feature_sets_and_labels( "QuickStartMagicSignal.csv",
																	  "QuickStartMagicBackground.csv", 
																	  num_of_vars=5, 
																	  test_size=0.3, 
																	  file_size=0.999, 
																	  var_set=[0,1,2,3,4,5,6,7,8,9] )
layer_vector = [10, 16, 9, 1]
network = jsdml.nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
#	Now we train the network on the training data and evaluate the testing data
network.train_network( train_x, train_y, num_epochs=100, batch=100 )
results = network.evaluate_network( test_x, test_y )
#	Now we evaluate the mutual information on the input and output
train_y = [[train_y[i]] for i in range(len(train_y))]
mutual = jsdml.mi(network.normalize_data(train_x),train_y,k=1)
signal, background = network.split_binary_results( results )
new_mutual = network.network_output_jsd(signal, background, neighbors=3)
print "MI before: ", mutual
print "MI after:  ", new_mutual
#	This section generates the ROC curves and calculates the area under curve	
network.plot_network_output(results, symmetric=False)



#	Example 3 - Breast Cancer Data
train_x, train_y, test_x, test_y = jsdml.create_feature_sets_and_labels( "QuickStartBCSignal.csv",
																	  "QuickStartBCBackground.csv", 
																	  num_of_vars=30, 
																	  test_size=0.1, 
																	  file_size=0.999 )
layer_vector = [30, 50, 15, 1]
network = jsdml.nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
#	Now we train the network on the training data and evaluate the testing data
network.train_network( train_x, train_y, num_epochs=100, batch=100 )
results = network.evaluate_network( test_x, test_y )
#	Now we evaluate the mutual information on the input and output
train_y = [[train_y[i]] for i in range(len(train_y))]
mutual = jsdml.mi(network.normalize_data(train_x),train_y,k=1)
signal, background = network.split_binary_results( results )
new_mutual = network.network_output_jsd(signal, background, neighbors=3)
print "MI before: ", mutual
print "MI after:  ", new_mutual
#	This section generates the ROC curves and calculates the area under curve	
network.plot_network_output(results, symmetric=False)



#	Example 3 - Breast Cancer Data Average MI
train_x, train_y, test_x, test_y = jsdml.create_feature_sets_and_labels( "QuickStartBCSignal.csv",
																	  "QuickStartBCBackground.csv", 
																	  num_of_vars=30, 
																	  test_size=0.01, 
																	  file_size=0.999 )
JSD_list = []
layer_vector = [30, 50, 15, 1]
num_samples = len(train_x)-1
num_random_samples = int(len(train_x)/2.0)
network = jsdml.nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
network.find_normalization_parameters(train_x)
train_x = network.normalize_data(train_x)
for i in range(100):
	#	Now randomly sample the train_x data set and calculate MI
	samples = np.random.randint(num_samples, size=num_random_samples)
	temp_train_x = [train_x[j] for j in samples]
	temp_train_y = [[train_y[j]] for j in samples]
	JSD_list.append(jsdml.mi(temp_train_x,temp_train_y,k=1))
	print "MI for sample set ",i,": ",JSD_list[i]
avg_JSD = sum(JSD_list)/100
print "MI average: ", avg_JSD


'''

#	Blank Code
train_x, train_y, test_x, test_y = jsdml.create_feature_sets_and_labels( "",
																	  "", 
																	  num_of_vars=, 
																	  test_size=0.3, 
																	  file_size=0.999, 
																	  var_set=[] )
layer_vector = []
network = jsdml.nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
#	Now we train the network on the training data and evaluate the testing data
network.train_network( train_x, train_y, num_epochs=100, batch=100 )
results = network.evaluate_network( test_x, test_y )
#	Now we evaluate the mutual information on the input and output
train_y = [[train_y[i]] for i in range(len(train_y))]
mutual = jsdml.mi(network.normalize_data(train_x),train_y,k=1)
signal, background = network.split_binary_results( results )
new_mutual = network.network_output_jsd(signal, background, neighbors=3)
print "MI before: ", mutual
print "MI after:  ", new_mutual
#	This section generates the ROC curves and calculates the area under curve	
network.plot_network_output(results, symmetric=False)

'''