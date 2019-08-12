import random
import cv2
import numpy as np

class Translation:
	
	def __init__(self, classes=[], rate=1, max_x=10, max_y=10):
		self.classes = classes
		self.rate = rate
		self.max_x = max_x
		self.max_y = max_y
	

	def __call__(self, image_label):
		image, label = image_label
		rows, cols, channels = image.shape
		images = []
		labels = []
		if label in self.classes:
			for transform in range(self.rate):
				tx, ty = random.random() * (self.max_x+1) * 2 - self.max_x, random.random()* (self.max_y+1) * 2 - self.max_y
				translation = np.float32([[1, 0, tx], [0, 1, ty]])
				dst = cv2.warpAffine(image, translation, (cols, rows))
				images.append(dst)
				labels.append(label)
		return (images, labels)
			
