#		JSD/Neural Network Implementation										             written by  Nicholas Carrara	7/19/17
#												  (and including routines by Greg Ver Steeg (https://github.com/gregversteeg/NPEET))
'''
	This is an implementation of a neural network using Keras and for calculating the Jensen-Shannon divergence on a continuous
	data set in N dimensions to compare with the neural network output which is typically one dimensional.  An example is included 
	in the main section of this code.  

	------------------------------------------------- Jensen-Shannon Divergence --------------------------------------------------  

		The JSD is a measure of inherent separation between two probability distributions and can be defined in several ways;

		1). JSD[p(x),q(x)] = S[pi_p * p(x) + pi_q * q(x)] - pi_p * S[p(x)] - pi_q * S[q(x)],      where pi_p + pi_q = 1
		2).                = I[x,theta],       where theta is a binary variable indicating either p(x) or q(x)

		S[p(x)] is the continuous/discrete Shannon entropy of p(x).  For more information see the documentation.

		This implementation calculates the JSD in both ways, borrowing some code from a package called NPEET for the mutual
		information version and a discrete form in one dimension which bins the sample distribution and calculates the
		Shannon entropies in (1).  


	---------------------------------------------- Neural Network Implementation -------------------------------------------------
	
		We use Keras with the backend set to tensorflow in this implementation.  




'''
######################################################
#	Required Packages
import numpy as np
import random
import scipy.spatial as ss
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import argparse
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn import preprocessing
import os
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
import keras
import csv
######################################################


##################################################################################################################################
################################################# - Auxiliary Functions - ########################################################

#	This function generates training and testing data from files specified by signal and background
def create_feature_sets_and_labels( signal, background, num_of_vars=1, test_size = 0.1, file_size=1.0, var_set=0.0 ):
	#	If the set of variables to use isn't specified, then we take the first N elements up to the num_of_vars
	if ( var_set == 0.0 ):
		var_set = [i for i in range(0, num_of_vars)]
	#	Now just read in all of the data from the two specified files
	with open( signal, 'rb' ) as sig_file:
		signal_list = [[float(row[i]) for i in var_set] for row in csv.reader(sig_file, delimiter=',')]
		sig_file.close()
	with open( background, 'rb' ) as back_file:
		background_list = [[float(row[i]) for i in var_set] for row in csv.reader(back_file, delimiter=',')]
		back_file.close()
	#	We only want to keep the amount of data specified by file_size.  We will take a random set of points from each
	#	of the specified files making sure that the ratio of signal/background = 1
	data = list()
	random.shuffle( signal_list )
	random.shuffle( background_list )
	signal_length = len( signal_list )
	background_length = len( background_list )
	temp_back_events = background_length * file_size
	temp_sig_events = signal_length * file_size
	if ( temp_back_events < temp_sig_events ):
		temp_events = temp_back_events
	elif ( temp_sig_events < temp_back_events ):
		temp_events = temp_sig_events
	else:
		temp_events = temp_sig_events
	if ( signal_length < temp_back_events ):
		background_list = list( background_list[:][-signal_length:] )
	elif ( background_length < temp_sig_events ):
		signal_list = list( signal_list[:][-background_length:] )
	#	Now that we've trunkated the lists according to the specified file_size, we populate a data set with the
	#	value of each variable (signal_list[i] or background_list[i]) and the answer ([1.0] or [-1.0])
	#	1.0 and -1.0 are arbitrary assignments to the class label, but are used here because of the choice of
	#	activation function built into the network (tanh(x))
	for k in range( len( signal_list ) ):
		data.append( [ list( signal_list[k] ), 1.0] )
	for k in range( len( background_list ) ):
		data.append( [ list( background_list[k] ), -1.0] )
	#	Once the list is populated, we shuffle again and separate the list into a training set whose length is specified 
	#	by (1 - test_size) and a testing set whose length is specified by test_size
	random.shuffle( data )
	sample_ratio = len( data ) - int( file_size * len( data ) )
	data = np.array( data[:-sample_ratio] )
	testing_ratio = int( test_size * len( data ) )
	train_x = list( data[:, 0][:-testing_ratio] )
	train_y = list( data[:, 1][:-testing_ratio] )
	test_x = list( data[:, 0][-testing_ratio:] )
	test_y = list( data[:, 1][-testing_ratio:] )
	print('Created Training and Testing Data from files; %s and %s' % (signal, background))
	print('With testing ratio = %s' % (test_size))
	#	Now for some garbage collection just to be safe, and then return the training and testing lists
	del signal_list, background_list
	return train_x,train_y,test_x,test_y

# Creates the acceptance/rejection cuts from the output data of the network
def acceptance_rejection_cuts( signal, background, cut_density=0.1, symmetric_signal=True ): 
    # We make cuts first
    num_signal_points = len( signal )
    num_background_points = len( background )
    #	The number of default points is 100, but any number can be specified by changing the value of cut_density
    cut_points = int( 1001 * cut_density )
    cut_point = 1.0
    step_size = 2.0 / ( cut_points - 1 )
    #	We must first normalize the data, by finding the max and min of both lists and dividing by the range.
    max_value = np.amax( np.concatenate( ( signal, background ), axis=0 ) )
    min_value = np.amin( np.concatenate( ( signal, background ), axis=0 ) )
    value_ranges = max_value - min_value
    signal[:] /= value_ranges - min_value
    background[:] /= value_ranges - min_value
	#	acceptance and rejection list initialization
    signal_efficiency = []
    background_efficiency = []															

    #	Now we find a set of symmetric cuts on the signal
    sorted_signal = np.sort( signal )
    sorted_signal = sorted_signal[::-1]														
    cut_interval = int( len( signal ) / cut_points )
    #	Finding even percentages of signal
    sorted_cut_points = [sorted_signal[i * cut_interval] for i in range( cut_points )]		
    #	If it is desired to have the cuts at equal percentages of signal acceptance, the symmetric_signal will be 
    #	set to true, which is the default option
    if ( symmetric_signal == True ):
    	for i in range( len( sorted_cut_points ) ):
    		signal_passed = 0.0
    		background_passed = 0.0
    		for j in range( num_signal_points ):
    			if signal[j] > sorted_cut_points[i]:
    				signal_passed += 1.0
    		signal_efficiency.append( signal_passed / num_signal_points )
    		for j in range( num_background_points ):
    			if background[j] > sorted_cut_points[i]:
    				background_passed += 1.0
    		background_efficiency.append( background_passed / num_background_points )
    elif ( symmetric_signal == False ):
		for i in range(0,cut_points):												
			signal_passed = 0.0
			background_passed = 0.0
			for j in range( num_signal_points ):
				if signal[j] > cut_point:
					signal_passed += 1.0
			signal_efficiency.append( signal_passed / num_signal_points )
			for j in range( num_background_points ):
				if background[j] > cut_point:
					background_passed += 1.0
			background_efficiency.append( background_passed / num_background_points )
			cut_point = cut_point - step_size
    signal_rejection = [1.0 - signal_efficiency[i] for i in range(len(signal_efficiency))]
    background_rejection = [1.0 - background_efficiency[i] for i in range(len(background_efficiency))]

    return np.array( signal_efficiency ), np.array( signal_rejection ), np.array( background_efficiency ), np.array( background_rejection )

#	This is a modified function for the average digamma evaluation which makes use of the fact that the answer space is discrete
def avgdigamma2(points,dvec):
  	N = len(points)
  	num_sig = -1.0
  	num_back = -1.0
  	for i in range(len(points)):
  		if (points[i][0] > 0.0):
  			num_sig += 1.0
  		else:
  			num_back += 1.0
  	avg = 0.0
  	for i in range(len(points)):
  		dist = dvec[i]
  		if (dist < 2.0):
  			if (points[i][0] > 0.0):
  				avg += digamma(num_sig) / N
  				#print('Computing MI for point %s with %s total points; %s' % (i,N,num_sig))
  			else:
  				avg += digamma(num_back) / N
  				#print('Computing MI for point %s with %s total points; %s' % (i,N,num_back))
  		else:
  			avg += digamma(N) / N
  	return avg

#	This function calculates a one-dimensional JSD by histograming the sample and using the definition of the Shannon entropy
def discrete_JSD( signal, background, num_bins=50 ):
	signal = [signal[i][0] for i in range(len(signal))]
	background = [background[i][0] for i in range(len(background))]
	joint = signal + background 
	maximum = max(joint)
	minimum = min(joint)
	bin_range = maximum - minimum
	bin_width = bin_range / num_bins
	bin_ranges = np.arange(minimum, maximum, bin_width)
	sig_prob = list()
	back_prob = list()
	joint_prob = list()
	for i in range(len(bin_ranges) - 1):
		temp_signal = 0
		temp_background = 0
		temp_joint = 0
		for j in range(len(signal)):
			if (signal[j] >= bin_ranges[i] and signal[j] < bin_ranges[i+1]):
				temp_signal += 1
		for j in range(len(background)):
			if (background[j] >= bin_ranges[i] and background[j] < bin_ranges[i+1]):
				temp_background += 1
		for j in range(len(joint)):
			if (joint[j] >= bin_ranges[i] and joint[j] < bin_ranges[i+1]):
				temp_joint += 1
		sig_prob.append(float(temp_signal) / len(signal))
		back_prob.append(float(temp_background) / len(background))
		joint_prob.append(float(temp_joint) / len(joint))
	sig_entropy = 0.0
	back_entropy = 0.0
	joint_entropy = 0.0
	for i in range(len(sig_prob)):
		if (sig_prob[i] != 0):
			sig_entropy -= sig_prob[i] * np.log2(sig_prob[i])
	for i in range(len(back_prob)):
		if (back_prob[i] != 0):
			back_entropy -= back_prob[i] * np.log2(back_prob[i])
	for i in range(len(joint_prob)):
		if (joint_prob[i] != 0):
			joint_entropy -= joint_prob[i] * np.log2(joint_prob[i])
	print("Signal Entropy:     %s " % (sig_entropy))
	print("Background Entropy: %s " % (back_entropy))
	print("Joint Entropy:      %s " % (joint_entropy))
	JSD = (joint_entropy - .5*(sig_entropy + back_entropy))
	print("JSD:                %s " % (JSD))
	return JSD


###############################################		Variations of NPEET code 	################################################
#		For more information see the documentation.  NPEET is available at; https://www.isi.edu/~gregv/npeet.html			   #

#	This function is for continuous x's and discrete y's
def mi(x,y,k=3,base=2):
  	""" Mutual information of x and y
    x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
    if x is a one-dimensional scalar and we have four samples
  	"""
  	assert len(x)==len(y), "Lists should have same length"
  	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  	intens = 1e-10 #small noise to break degeneracy, see doc.
  	x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  	y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
  	points = zip2(x,y)
  	#Find nearest neighbors in joint space, p=inf means max-norm
  	tree = ss.cKDTree(points)
  	dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
  	a,b,c,d = avgdigamma(x,dvec), avgdigamma2(y,dvec), digamma(k), digamma(len(x)) 
  	return (-a-b+c+d)/log(base)

#	This function is for both x and y continuous
def mi2(x,y,k=3,base=2):
  	""" Mutual information of x and y
    x,y should be a list of vectors, e.g. x = [[1.3],[3.7],[5.1],[2.4]]
    if x is a one-dimensional scalar and we have four samples
  	"""
  	assert len(x)==len(y), "Lists should have same length"
  	assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
  	intens = 1e-10 #small noise to break degeneracy, see doc.
  	x = [list(p + intens*nr.rand(len(x[0]))) for p in x]
  	y = [list(p + intens*nr.rand(len(y[0]))) for p in y]
  	points = zip2(x,y)
  	#Find nearest neighbors in joint space, p=inf means max-norm
  	tree = ss.cKDTree(points)
  	dvec = [tree.query(point,k+1,p=float('inf'))[0][k] for point in points]
  	a,b,c,d = avgdigamma(x,dvec), avgdigamma(y,dvec), digamma(k), digamma(len(x)) 
  	return (-a-b+c+d)/log(base)

	#####INTERNAL FUNCTIONS

def avgdigamma(points,dvec):
  	#This part finds number of neighbors in some radius in the marginal space
  	#returns expectation value of <psi(nx)>
  	N = len(points)
  	tree = ss.cKDTree(points)
 	avg = 0.
  	for i in range(N):
  		dist = dvec[i]
  		#subtlety, we don't include the boundary point, 
  		#but we are implicitly adding 1 to kraskov def bc center point is included
  		num_points = len(tree.query_ball_point(points[i],dist-1e-15,p=float('inf'))) 
  		avg += digamma(num_points)/N
  	return avg

def zip2(*args):
  	#zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
  	#E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
  	return [sum(sublist,[]) for sublist in zip(*args)]

###########################################		END OF Variations of NPEET code 	##############################################


##################################################################################################################################
################################################# - Neural Network Class - #######################################################

class nnet:

	def __init__( self, layer_vector, learning_rate=0.1, decay_value=1e-6, momentum_value=0.9, nest=True ):
		
		#	Layer vector is a vector of node numbers.  For a network with 4 input nodes, 10 hidden nodes and 1 output node 
		#	layer_vector should read [4,10,1]
		self.layer_vector = layer_vector
		#	These will be the network parameters set after training		
		self.parameters = None	
		#	The normalization parameters set by the training data				
		self.normalization = []					
		self.model = Sequential()

		#	Now we specify the model from layer_vector
		self.model.add( Dense( layer_vector[1], input_dim=layer_vector[0], init='uniform', bias=True ) )	
		self.num_additions = 0
		for i in range( 1, len( layer_vector ) - 1 ):
			#	This "layer" object applies the activation from the output of the previous
			self.model.add( Activation( 'tanh' ) )
			#	Adding the next layer															
			self.model.add( Dense( layer_vector[i+1], init='uniform', bias=True ) )							
			self.num_additions += 2
		self.model.add( Activation( 'tanh' ) )
		self.output_function = K.function( [self.model.layers[0].input], [self.model.layers[self.num_additions + 1].output] )
		#	Stochastic Gradient Descent
		sgd = SGD( lr=learning_rate, decay=decay_value, momentum=momentum_value, nesterov=nest )
		#	This compiles the network			
		self.model.compile( loss= 'mean_squared_error', optimizer=sgd, metrics=['accuracy'] )				

	def get_parameters( self ):
		return self.parameters

	#	This function sets the parameters from a given list
	def set_parameters( self, param ):
		self.model.set_weights( param )
		self.parameters = param

	def get_normalization( self ):
		return self.normalization

	def set_normalization( self, normal ):
		self.normalization = normal

	def find_normalization_parameters( self, data ):
		#	Scaling is as follows; Do standardization on all the variables; x' = (x - mu)/sigma
		#	Then apply the sigmoid function; x'' = 1/(1 + e^(-x))
		#	Then search for nearest neighbors and scale by <RMS>
		self.normalization = []
		
		means = np.asarray(data).mean(axis=0)
		sigmas = np.asarray(data).std(axis=0)

		#	Now make a copy and apply the standardization
		temp_data = np.copy(data)
		for i in range( len( temp_data[0]) ):
			for j in range( len( temp_data ) ):
				#print temp_data[j][i]
				temp_data[j][i] = ( temp_data[j][i] - means[i] ) / sigmas[i]

		#	Now  apply the sigmoid function
		for i in range( len( temp_data[0]) ):
			for j in range( len( temp_data ) ):
				#print temp_data[j][i]
				temp_data[j][i] = 1 / (1 + np.exp(-temp_data[j][i]))

		total_num = 1000
		#	Now find the nearest neighbor <RMS>
		squares = np.zeros(len(temp_data[0]))
		num_points = np.random.randint(0,len(temp_data),total_num)
		temp_data_2 = [temp_data[num_points[i]] for i in range(len(num_points))]
		for i in range(total_num):
			temp_neighbor = 0
			temp_distance = 10e6
			temp_point = temp_data_2[i]
			for j in range(len(temp_data_2)):
				if i != j:
					temp_dist = 0.0
					for l in range(len(temp_point)):
						temp_dist += np.power((temp_point[l] - temp_data_2[j][l]),2.0)
					if np.sqrt(temp_dist) <= temp_distance:
						temp_distance = np.sqrt(temp_dist)
						temp_neighbor = j
			#print "Found nearest neighbor for point ", i, " at point ", j
			for l in range(len(squares)):
				squares[l] += (np.power((temp_data_2[i][l] - temp_data_2[temp_neighbor][l]),2.0) / total_num)
		max_square = squares[0]
		for i in range(len(squares)):
			if (squares[i] >= max_square):
				max_square = squares[i]
		squares = squares / max_square
		
		#	Now apply the nearest neighbor <RMS> to find a new mean
		for i in range( len( temp_data[0]) ):
			for j in range( len( temp_data ) ):
				#print temp_data[j][i]
				temp_data[j][i] = temp_data[j][i] / np.sqrt(squares[i])
		num_points = np.random.randint(0,len(temp_data),total_num)
		temp_data_2 = [temp_data[num_points[i]] for i in range(len(num_points))]
		new_means = np.asarray(temp_data_2).mean(axis=0)

		#	Now save these parameters 
		for i in range(len(squares)):
			self.normalization.append( [ means[i], sigmas[i], np.sqrt(squares[i]), new_means[i] ] )

	#	Trying to do this with a list comprehension is tricky
	def normalize_data( self, data ):
		for i in range( len( self.normalization) ):
			for j in range( len( data ) ):
				#print data[j][i]()
				data[j][i] = ((1 / (1 + np.exp(-((data[j][i] - self.normalization[i][0]) / self.normalization[i][1])))) / self.normalization[i][2]) - self.normalization[i][3]
				#data[j][i] = (1 / (1 + np.exp(-( data[j][i] - self.normalization[i][0] ) / self.normalization[i][1]))) / self.normalization[i][2]# / self.normalization[i][2]
		return data

	#	This function trains the network on specified training data
	def train_network( self, training_data, training_answer, num_epochs=1, batch=256 ):
		train_data = np.copy( training_data )
		#	Anytime we are training a network, we must renormalize according to the data
		self.find_normalization_parameters( training_data )													
		train_data = self.normalize_data( train_data )		
		#	The training session	
		self.model.fit( train_data, training_answer, nb_epoch=num_epochs, batch_size=batch )	
		#	Saves the weights from training to the parameters attribute			
		self.parameters = self.model.get_weights()															

	#	This function evaluates test data against the trained network
	def evaluate_network( self, testing_data, testing_answer, score_output=True ):
		test_data = np.copy( testing_data )		
		#	We don't want to normalize the actual testing data, only a copy of it														
		test_data = self.normalize_data( test_data )
		if ( score_output == True ):
			score = self.model.evaluate( test_data, testing_answer, batch_size=100 )		
			#	Prints a score for the network based on the training data				
			print('Score: %s' % ( score ) )
		activations = self.output_function( [test_data] )
		return [ [activations[0][i][0], testing_answer[i]] for i in range( len( testing_answer ) ) ]

	#	This takes the network output and splits them into separate signal and background variables 
	def split_binary_results( self, results ):
		signal = []
		background = []
		for i in range( len( results ) ):
			if ( results[i][1] == -1.0 ):
				background.append( results[i][0] )
			else:
				signal.append( results[i][0] )
		return signal, background

	def network_output_jsd( self, signal, background, neighbors=3):
		new_signal = [[signal[i]] for i in range(len(signal))]
		new_background = [[background[i]] for i in range(len(background))]
		new_ans1 = [[1.0] for s in range(len(signal))]
		new_ans2 = [[-1.0] for s in range(len(background))]
		new_ans = new_ans1 + new_ans2
		new_data = new_signal + new_background
		return mi(new_data, new_ans,k=neighbors)

	#	We plot three things, a histogram of the network output and two ROC curves (and accept/reject and accept/accept for signal and background)
	def plot_network_output( self, results, save_cuts=True, symmetric=True ):
		keras_signal, keras_background = self.split_binary_results( results )
		keras_signal_acc, keras_signal_rej, keras_background_acc, keras_background_rej = acceptance_rejection_cuts( keras_signal,
																											        keras_background,
																											        symmetric_signal=symmetric )
		keras_sig_acc_back_rej_AUC = self.area_under_ROC_curve( keras_signal_acc, keras_background_rej )
		if ( save_cuts == True ):
			cut_values = [ [ keras_signal_acc[i], keras_background_acc[i] ] for i in range( len( keras_signal_acc ) ) ]
			with open( 'cut_values.csv', 'w' ) as cut_file:
				writer = csv.writer( cut_file )
				writer.writerows( cut_values )

		print( 'AUC: Background Rej. vs. Signal Acc.;' )
		print( 'Keras/Tensorflow: %s' % ( keras_sig_acc_back_rej_AUC ) )
		#	Now plotting our results
		plt.figure(1)
		plt.hist( keras_signal, 100, normed='True', alpha=0.5, facecolor='blue', label='Signal', hatch="/")
		plt.hist( keras_background, 100, normed='True', alpha=0.5, facecolor='red', label='Background', hatch="/")
		plt.legend( loc='upper right' )
		plt.title( 'Keras/TensorFlow test data histogram' )
		plt.savefig( 'nn_hists.png' )

		plt.figure(2)
		plt.plot( keras_signal_acc, keras_background_acc, linestyle='None' )
		plt.scatter( keras_signal_acc, keras_background_acc, color='k', label='keras/tensorflow' )
		plt.xlabel( 'Signal Acceptance' )
		plt.ylabel( 'Background Acceptance' )
		plt.yscale( 'log' )
		plt.xscale( 'log' )
		plt.legend( loc='upper left', shadow=True, title='Legend', fancybox=True )
		plt.grid(True)
		plt.title( 'Bkd Acc vs. Sig Acc' )
		plt.ylim(0.0,1.0)
		plt.xlim(0.0,1.0)

		plt.figure(3)
		plt.plot( keras_signal_acc, keras_background_rej, linestyle='None' )
		plt.scatter( keras_signal_acc, keras_background_rej, color='k', label='keras/tensorflow' )
		plt.xlabel( 'Signal Acceptance' )
		plt.ylabel( 'Background Rejection' )
		plt.legend( loc='upper left', shadow=True, title='Legend', fancybox=True )
		plt.grid(True)
		plt.title( 'Bkd Rej vs. Sig Acc' )
		plt.ylim(0.0,1.0)
		plt.xlim(0.0,1.0)
		
		plt.show()

	#This can get passed two CDF or inverse CDF values to calculate the total area under the receiver-operator-characteristic
	def area_under_ROC_curve( self, signal, background ):
		num_data_points = len( signal )
		area_under_curve = 0.0
		for i in range( num_data_points - 1 ):
			area_under_curve += background[i] * ( signal[i+1] - signal[i] )
		return area_under_curve

	#	This saves the network parameters to a file which can be restored later
	def save_network_params_to_file( self, file ):
		with open( file, 'w' ) as param_file:
			writer = csv.writer( param_file )
			writer.writerows( self.parameters )

	#	This saves the network output along with the correspond value in the variable space
	def save_network_score_to_file( self, data, results, file ):
		#	Need to make a new list with [ [ data, result[0] ] ] and then write that to file
		output_list = list()
		for i in range( len( data ) ):
			output_list.append( np.concatenate( ( data[i], [results[i][0]] ) ) )
		with open( file, 'w' ) as ntuple_file:
			writer = csv.writer( ntuple_file )
			writer.writerows( output_list )
		print('Wrote data to file; %s' % ( file ) )



# If this file is being run it will go through the SUSY example with the 8 low-level variables.
if __name__ == "__main__":
	
	#	First we import all of the data from the signal and background files for the first eight low level variables.
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels( "SUSYSignal.csv",
																	   "SUSYBackground.csv", 
																	   num_of_vars=8, 
																	   test_size=0.001, 
																	   file_size=0.999, 
																	   var_set=[0,1,2,3,4,5,6,7] )
	num_points = len(train_x)
	skip_val = len(train_x) / 10
	starting_val = 0
	ending_val = skip_val - 1
	JSD_list = list()
	AUC_list = list()

	#	Here we break up the data into ten random subsets and run the full chain on each, getting an average and standard deviation for
	#	all calculated values.
	for j in range(0,10):
		temp_scores = list()
		layer_vector = [8, 14, 7, 1]
		network = nnet( layer_vector, learning_rate=0.05, decay_value=1e-5 )
		temp_train_x = [train_x[i] for i in range(starting_val,ending_val)]
		temp_train_y = [train_y[i] for i in range(starting_val,ending_val)]
		#	Now separate the training and testing by a 70 / 30 random split
		testing_ratio = int( .3 * skip_val )
		new_train_x = list( temp_train_x[:-testing_ratio] )
		new_train_y = list( temp_train_y[:-testing_ratio] )
		new_test_x = list( temp_train_x[-testing_ratio:] )
		new_test_y = list( temp_train_y[-testing_ratio:] )
		#	Now train the network on the training set
		network.train_network( new_train_x, new_train_y, num_epochs=1, batch=100 )
		#	Then, evaluate the network on the testing results
		results = network.evaluate_network( new_test_x, new_test_y )
		#	We must break up the testing results into separate signal/background so that we can calculate the mutual information
		signal, background = network.split_binary_results( results )	
		temp_train_y = [[temp_train_y[i]] for i in range(len(temp_train_y))]
		#	Getting the area under curve for background rejection versus signal efficiency
		sig_eff, sig_rej, bkd_eff, bkd_rej = acceptance_rejection_cuts(signal, background)
		AUC = network.area_under_ROC_curve(sig_eff,bkd_rej)
		#	Now calculating the before and after mutual information
		mutual = mi(network.normalize_data(temp_train_x),temp_train_y,k=1)
		new_mutual = network.network_output_jsd(signal, background, neighbors=1)
		#	Now we save all the values we've calculated for this set of sample data
		temp_scores.append(mutual)
		temp_scores.append(new_mutual)
		JSD_list.append(temp_scores)
		AUC_list.append(AUC)
		print "trial:    ", j
		print "MI before:", mutual
		print "MI after: ", new_mutual
		print "AUC:      ", AUC
		starting_val += skip_val
		ending_val += skip_val

	#	Now take the mean and standard deviation of the JSD's and AUC's and print them to screen
	before_list = [JSD_list[i][0] for i in range(len(JSD_list))]
	after_list = [JSD_list[i][1] for i in range(len(JSD_list))]
	before_mean = sum(before_list) / 10
	after_mean = sum(after_list) / 10

	before_std = np.std(np.array(before_list))
	after_std = np.std(np.array(after_list))

	AUC_mean = sum(AUC_list) / 10
	AUC_std = np.std(np.array(AUC_list))

	print "MI before mean:     ",before_mean
	print "MI before std:      ",before_std
	print "MI after mean:      ",after_mean
	print "MI after std:       ",after_std
	print "eff/rej AUC mean:   ",AUC_mean
	print "eff/rej AUC std:    ",AUC_std