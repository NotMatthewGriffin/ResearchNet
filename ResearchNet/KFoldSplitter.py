import random

class KFoldSplitter:

    '''
    Creates the KFold Splitter to load data for kfold cross validation
    specify a seed to ensure that the randomly selected folds are the same each time
    '''
    def __init__(self, 
                 dataLoader, 
                 k=2,
                 seed=None,
                 resampler=None,
                 cropper=None,
                 preprocessor=[],
                 augmentations=[],
                 worst_case_augmentations=[], 
                 normalization=None,
                 starting_size=(40, 40),
                 ending_size=(40, 40)):
        self.seed = seed
        self.dataLoader = dataLoader
        self.k = k
        self.augmentations = augmentations
        self.worst_case_augmentations = worst_case_augmentations
        self.resampler = resampler if resampler else lambda x, y : x
        self.preprocessing = preprocessor
        self.starting_size = starting_size
        self.ending_size = ending_size
        # if there is a cropper apply it right before normalization
        if cropper:
            self.preprocessing.append(cropper)
        # if there is a normalizer then apply it last
        if normalization:
            self.preprocessing.append(normalization)

    '''
    load the nth fold of k as the training set and all other folds as the training set
    '''
    def load_fold_n(self, n):	
        random.seed(self.seed)
        all_target_indices = list(range(len(self.dataLoader.files_to_read)))
        # shuffle all the target indices
        random.shuffle(all_target_indices)

	# index indics is used to assign the folds out of k
        index_indices = list(range(len(all_target_indices)))
	
	# assign training and testing indices
        training = []
        testing = []
        for index in index_indices:
            if (index % self.k) == (n % self.k):
                testing.append(all_target_indices[index])
            else:
                training.append(all_target_indices[index])
        training = self.perform_resample(training)
        return {"training":self.dataLoader.load_data(training,
                                                     preprocess = self.preprocessing, 
                                                     augmentation = self.augmentations, 
                                                     starting_size = self.starting_size, 
                                                     ending_size = self.ending_size), 
                "testing":self.dataLoader.load_data(testing, 
                                                    preprocess = self.preprocessing,
                                                    starting_size = self.starting_size,
                                                    ending_size = self.ending_size)
               }

    def worst_case_test_data(self, n):
        # must seed the same or the folds will be different
        random.seed(self.seed)
        all_target_indices = list(range(len(self.dataLoader.files_to_read)))
        # shuffle all the target indices
        random.shuffle(all_target_indices)

	# index indics is used to assign the folds out of k
        index_indices = list(range(len(all_target_indices)))
	
	# assign training and testing indices
        training = []
        testing = []
        for index in index_indices:
            if (index % self.k) == (n % self.k):
                testing.append(all_target_indices[index])
            else:
                training.append(all_target_indices[index])
        training = self.perform_resample(training)
        return {"testing":self.dataLoader.load_data(testing, 
                                         preprocess = self.preprocessing,
                                         augmentation = self.worst_case_augmentations,
                                         starting_size = self.starting_size,
                                         ending_size = self.ending_size)}

		
		
    def perform_resample(self, training_data):	
        # here training data is not actually loaded yet
        # data is just an index representing something that should be loaded
        return self.resampler(training_data, self.dataLoader.load_labels(training_data))


