import random
import cv2

no_rotation = 0

class Scale:

	def __init__(self, classes=[], rate=1, min_scale=0.8, max_scale=2):
		self.classes = classes
		self.rate = rate
		self.max_scale = max_scale
		self.min_scale = min_scale

	
	def __call__(self, image_label):
		image, label = image_label
		rows, cols, channels = image.shape
		images = []
		labels = []
		if label in self.classes:
			for transform in range(self.rate):
				ran_scale = (random.random() * (self.max_scale-self.min_scale)) + self.min_scale
				scaling = cv2.getRotationMatrix2D((cols//2, rows//2), no_rotation, ran_scale)
				dst = cv2.warpAffine(image, scaling, (cols, rows))
				images.append(dst)
				labels.append(label)
		return (images, labels)
