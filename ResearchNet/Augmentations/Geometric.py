import random
import cv2
import numpy as np

class Geometric:
	
	def __init__(self, classes=[], rate=1, max_x=10, max_y=10, max_angle=30, min_scale=0.8, max_scale=2.0):
		self.classes = classes
		self.rate = rate
		self.max_x = max_x
		self.max_y = max_y
		self.max_angle = max_angle
		self.min_scale = min_scale
		self.max_scale = max_scale

	def __call__(self, image_label):
		image, label = image_label
		rows, cols, channels = image.shape
		images = []
		labels = []
		if label in self.classes:
			for transform in range(self.rate):
				rotation = random.random()*(2*self.max_angle)-self.max_angle
				scale = (random.random() * (self.max_scale-self.min_scale)) + self.min_scale
				tx, ty = random.random() * (self.max_x+1) * 2 - self.max_x, random.random()* (self.max_y+1) * 2 - self.max_y
				translation = np.float32([[1, 0, tx], [0, 1, ty]])
				rotation_scale = cv2.getRotationMatrix2D((cols//2, rows//2), rotation, scale)
				dst = cv2.warpAffine(image, translation, (cols, rows))
				dst2 = cv2.warpAffine(dst, rotation_scale, (cols, rows))
				images.append(dst2)
				labels.append(label)
		return (images, labels)
				
