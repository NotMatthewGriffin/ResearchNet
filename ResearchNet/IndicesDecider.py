import random

class IndicesDecider:
	
	def __init__(self, length_of_data, k, seed):
		self.length = length_of_data
		self.k = k
		self.seed = seed


	def get_indices(self, n):
		random.seed(self.seed)
		all_target_indices = list(range(self.length))
		# shuffle all the target indices
		random.shuffle(all_target_indices)
		
		# index indics is used to assign the folds out of k
		index_indices = list(range(len(all_target_indices)))
		
		# assign training and testing indices
		training = []
		testing = []
		for index in index_indices:
		    if (index % self.k) == (n % self.k):
		        testing.append(all_target_indices[index])
		    else:
		        training.append(all_target_indices[index])
		return { 'training':training, 'testing':testing }
