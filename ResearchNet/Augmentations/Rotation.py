import cv2
import random

class Rotation:
	
	def __init__(self, classes=[], rate=1, max_angle=30):
		self.classes = classes
		self.rate = rate
		self.max_angle = max_angle

	def __call__(self, image_label):
		image, label = image_label
		rows, cols, channels = image.shape
		images = []
		labels = []
		if label in self.classes:
			for transform in range(self.rate):
				transform = cv2.getRotationMatrix2D((cols//2,rows//2), random.random()*(2*self.max_angle)-self.max_angle, 1)
				dst = cv2.warpAffine(image, transform, (cols, rows))
				images.append(dst)
				labels.append(label)
		return (images, labels)
