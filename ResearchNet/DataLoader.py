import cv2
import numpy as np
from os import walk
from functools import reduce

class DataLoader:
    
    def __init__(self, dataset):
        self.path_to_dataset = dataset
        self.files_to_read = []
        for line in walk(dataset):
            self.files_to_read.extend([line[0]+'/'+a_file for a_file in line[2]])

    def load_data(self, indices=[], starting_size=(40, 40), ending_size=(40, 40), augmentation=[], preprocess=[]):
        if not indices:
            indices = list(range(len(self.files_to_read)))
        load_files = list(map(lambda x : self.files_to_read[x], indices))
        images = map(cv2.imread, load_files)
        if starting_size:
            images = list(map(lambda x : cv2.resize(x, starting_size), images))
        else:
            images = list(images)
        # apply the augmentations to all the loaded images and their associated labels
        labels = self.load_labels(indices=indices)
        if augmentation:
            labels = labels.tolist()
            for augmentation_step in augmentation:
                images_labels = map(augmentation_step, zip(images, labels))
                images_arrs, labels_arrs = zip(*images_labels)
                # reduce the nested arrays of images_arrs and labels_arrs to flat arrays
                reduced_images = reduce(lambda x, y : x + y, images_arrs, [])
                reduced_labels = reduce(lambda x, y : x + y, labels_arrs, [])
                images.extend(reduced_images)
                labels.extend(reduced_labels)
        # apply the preprocesing to all the loaded images only (labels will not be changed by this)
        if preprocess:
            for preprocess_step in preprocess:
                images = map(preprocess_step, images)
            images = list(images)
        if ending_size:
            images = np.array(list(map(lambda x: cv2.resize(x, ending_size), images)), dtype=np.float32)
        else:
            images = list(images)
            print('jagged input array')

        return {"images":images, "labels":np.array(labels, dtype=np.int64)}

    # if its important to load just the labels
    def load_labels(self, indices=[]):
        if not indices:
            indices = list(range(len(self.files_to_read)))
        load_files = list(map(lambda x : self.files_to_read[x], indices))
        labels = list(map(lambda x : 0 if 'H' in x else 1, load_files))
        return np.array(labels, dtype=np.int64)
