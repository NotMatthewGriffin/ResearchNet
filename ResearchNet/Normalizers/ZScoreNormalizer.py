import numpy as np

class ZScoreNormalizer:
	
	def __call__(self, x):
		mean = np.mean(x)
		std = np.std(x)
		return (x-mean)/std
