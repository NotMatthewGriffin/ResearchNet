import random
import numpy as np

class Upsampler:
    def __init__(self, class_to_upsample=[], rate=2, seed=1):
        self.up_sample_class = class_to_upsample
        self.rate = rate
        self.seed = seed

    def __call__(self, indices, labels):
        random.seed(self.seed)
        for a_class in self.up_sample_class:
            # find where the class is equal and duplicate rate-1 times
            to_duplicate = np.where(labels == a_class)
            for x in range(self.rate-1):
                indices.extend(list(map(lambda x : indices[x], to_duplicate[0])))
        random.shuffle(indices)
        return indices

