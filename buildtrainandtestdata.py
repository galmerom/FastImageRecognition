# -*- coding: utf-8 -*-
"""
This model creates 3 image generator data sets: Train, validation and Test.

The train and validation is part of the same dataset. The dataset is divided to 2' one that is large is the training
and the other is the validation. The training dataset is the only one that the model sees during training.

The test data set is a different set of data. During the fitting process this data is not even seen by the user, thus
to avoid over fitting of the model using the parameters.

CrtTrainValidAndTestGenerators() - Generates: Training set and validation set from the train_dataset path. The generated
                                  images are taken from different directories. Each directory name becomes the label
                                  of the image.
                                  The training set is also gets many augmented images to improve the training
                                  The test set only get the images preprocessed. No augmentation. This set also gets
                                  the labels from the directories names.


"""

import numpy as np

# from tensorflow.keras.applications import imagenet_utils
from keras.applications import imagenet_utils


def CrtTrainValidAndTestGenerators(TrainImageGenerator, TestImageGenerator, paths, ParamDic, rnd=1234,
                                   Path2AugmentImg=False):
    """
    Create training, validation and test datasets. Also returns the number of steps needed to go over every generator.

    :param TrainImageGenerator: ImageGenerator. The image generator object. Made by keras library. Used for training.
    :param TestImageGenerator: ImageGenerator. The image generator object. Made by keras library. Used for testing.
    :param paths: dictionary. Dictionary with directory names as keys and paths as values.
    :param ParamDic: dictionary. Dictionary of parameters entered by the user
    :param rnd: int. Random seed.
    :param Path2AugmentImg: bool. If True: training generator will save augmented photos
    :return: 2 tuples:  (Training,validation,test generators) (Training,validation,test step size)
                        First tuple are ImageGenerator type and the second tuple are integer type
    """

    train_generator = TrainImageGenerator.flow_from_directory(paths['train_dataset'],
                                                              target_size=(
                                                                  ParamDic['IMAGE_HEIGHT'], ParamDic['IMAGE_WIDTH']),
                                                              batch_size=ParamDic['BatchSize'],
                                                              class_mode='categorical',
                                                              shuffle=True,
                                                              subset='training',
                                                              # save_to_dir=augmentationPath,
                                                              seed=rnd)

    # In case Augmented images is requested then save the in 'augmented pictures'
    if Path2AugmentImg:
        train_generator.save_to_dir = paths['augmented pictures']

    valid_generator = TrainImageGenerator.flow_from_directory(paths['train_dataset'],
                                                              target_size=(
                                                                  ParamDic['IMAGE_HEIGHT'], ParamDic['IMAGE_WIDTH']),
                                                              batch_size=ParamDic['BatchSize'],
                                                              class_mode='categorical',
                                                              subset='validation',
                                                              seed=rnd)
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    # create the test generator
    test_generator = TestImageGenerator.flow_from_directory(paths['test_dataset'],
                                                            target_size=(
                                                                ParamDic['IMAGE_HEIGHT'], ParamDic['IMAGE_WIDTH']),
                                                            batch_size=ParamDic['BatchSize'],
                                                            shuffle=False,
                                                            class_mode='categorical')
    STEP_SIZE_Test = test_generator.n // test_generator.batch_size

    return (train_generator, valid_generator, test_generator), (STEP_SIZE_TRAIN, STEP_SIZE_VALID, STEP_SIZE_Test)


def preproc(x):
    """
    Use this function to preprocess an image before entering a keras model
    :param x: Image in np array format
    :return: Image after preprocess
    """
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    return x
