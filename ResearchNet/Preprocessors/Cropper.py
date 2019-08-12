class Cropper:

	def __init__(self, left=0, top=0, width=40, height=40):
		self.left = left
		self.top = top
		self.width = width
		self.height = height


	def __call__(self, image):
		return image[self.top:self.top+self.height, self.left:self.left+self.width]

