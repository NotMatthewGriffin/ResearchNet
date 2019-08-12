import numpy as np

class DecimalScalingNormalizer:

	def __call__(self, x):
		a_max = max(np.max(x), np.min(x)*-1)
		num_tens = len(str(int(a_max)))
		div = 10**num_tens
		return x/div
