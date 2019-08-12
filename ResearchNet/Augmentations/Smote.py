import numpy as np
import random

class Smote:

	def __init__(self, classes=[], rate=1, dataLoader=None, data_full_size=(40,40)):
		self.classes = classes
		self.rate = rate
		labels = dataLoader.load_labels()
		# load the data at the full size it will be loaded at
		self.data = [dataLoader.load_data(indices=np.where(labels == a_class)[0].tolist(), starting_size=data_full_size, ending_size=data_full_size)["images"] for a_class in classes]


	def __call__(self, image_label):
		image, label = image_label
		rows, cols, channels = image.shape
		images = []
		labels = []
		if label in self.classes:
			# 5 (from original smote paper) nearest neighbors for one point skip first result because that should be itself
			distance_arr = np.sum(np.square(image - self.data[self.classes.index(label)]), axis=tuple(x+1 for x in range(len(image.shape))))
			closest_indices = np.argsort(distance_arr)[1:5+1]
			for index in random.choices(closest_indices, k=self.rate):
				gap = self.data[self.classes.index(label)][index] - image
				adder = gap * np.random.uniform(0, 1, size=image.shape)
				synthetic = image.copy() + adder
				images.append(synthetic)
				labels.append(label)
		return (images, labels)

		
