import numpy as np

class MinMaxNormalizer:

	def __call__(self, x):
		a_max = np.max(x)
		a_min = np.min(x)
		the_range = a_max-a_min
		return (x-a_min)/the_range
