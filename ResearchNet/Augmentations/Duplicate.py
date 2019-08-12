class Duplicate:
	def __init__(self, classes=[], rate=1):
		self.classes = classes
		self.rate = rate

	def __call__(self, image_label):
		image, label = image_label
		rows, cols, channels = image.shape
		if label in self.classes:
			images = [image for image_duplicate in range(self.rate)]
			labels = [label for image_duplicate in range(self.rate)]
		return (images, labels)

