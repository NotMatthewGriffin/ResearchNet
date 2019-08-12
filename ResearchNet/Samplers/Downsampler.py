import random
import numpy as np

class Downsampler:
    def __init__(self, class_to_downsample=[], rate=2, seed=1):
        self.down_class = class_to_downsample
        self.rate = rate
        self.seed = seed

    def __call__(self, indices, labels):
        random.seed(self.seed)
        for a_class in self.down_class:
            all_removed = np.where(labels == a_class)[0]
            keep_set = set(all_removed[::self.rate])
            remove_set = sorted(list(set(all_removed).difference(keep_set)))
            for to_remove in remove_set[::-1]:
                del indices[to_remove]
        return indices
