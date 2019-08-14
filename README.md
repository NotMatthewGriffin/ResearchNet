# Set up to run
The easiest way to run this software is with the included Dockerfile. Simply build the Dockerfile and run the image. Copy the software to the container and run it interactively by attaching. Otherwise you can run the software any where that you have all the necessary dependencies. 

In order to run the software without the Docker container you will need the following dependencies installed:

* python3
* numpy
* opencv (cv2)
* pytorch
* torchvision

# Running the software
To run the software, navigate to the directory where the code is located and run the following command:

`python AugmentationTest.py foldNumber [-augmentationName augmentationAmount]`

where you replace fold\_number with the fold of testing you would like to perform. Replace augmentationName with the name of the augmentation you would like to use and augmentationAmount with the amount of augmentation you would like to apply. As configured the software only tests 20 folds, running with any higher number will not cause more folds to be used, in fact the argument will be treated as foldNumber % 20. augmentationName and augmentationAmount are optional but if one is included so must the other, additionally they must occur after the foldNumber.

The following augmentations are supported:

* translate
* rotate
* scale
* geometric
* encodeOpposite
* encodeSame
* decodeRandom
* duplicate
* justEncode
* smallTranslation

