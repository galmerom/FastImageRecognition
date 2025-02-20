# -*- coding: utf-8 -*-
"""
This module helps building a keras model based on resnet50 without the top

CreateSpecificModel() - Builds a Keras model based on resnet50 without top. The last 2 layers are average pooling
                        and dense layer with softmax.
PickLearnRateDecay() - This function allows the user to pick a learning rate decay.
                       Currently there are the following to choose from:
                       CosineDecay
                       CosineDecayRestarts
                       LinearCosineDecay
PickOptimizer() - This function allows the user to pick an optimizer.
                       Currently there are the following to choose from:
                       adam
                       Nadam
                       Adamax
                       SGD

FindWeights() - This function creates a dictionary to balance the weight of the different classes.
                It also allows to add extra weight to a specific class

ShowEpochHistory() = Show 2 charts, one that show the change in loss function and the other shows the accuracy
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers.schedules import CosineDecay, CosineDecayRestarts
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, SGD
from collections import Counter


def CreateSpecificModel(OptimizerObj, numClass=2, ModelWeights='imagenet', ImageShape=(224, 224, 3),
                        LossFunc='categorical_crossentropy', Metrics=['accuracy']):
    """
    Creates a keras model based on resnet50 with average pooling and simplex dense layer instead of the original top.

    :param OptimizerObj: Optimizer keras object
    :param numClass: int. Number of classes to identify
    :param ModelWeights: dictionary. A dictionary that contains each class and the relevant weights
    :param ImageShape: tuple. tuple used to define the image size (width,height,channels)
    :param LossFunc: string. The loss function desired
    :param Metrics: List of strings. Strings that contains the metrics desired
    :return: A keras model
    """
    resnet50_imagenet_model = ResNet50(include_top=False, weights=ModelWeights, input_shape=ImageShape)

    avgp = tf.keras.layers.GlobalAveragePooling2D()(resnet50_imagenet_model.output)
    fc2 = tf.keras.layers.Dense(numClass, activation='softmax', name="AddedDense2")(avgp)

    model = tf.keras.models.Model(inputs=resnet50_imagenet_model.input, outputs=fc2)

    model.compile(optimizer=OptimizerObj, loss=LossFunc, metrics=Metrics)

    return model


def PickLearnRateDecay(option, ParamDic={}):
    """
    Build a changing learning rate object.

    It gets the type of changing learning rate needed and the parameters that the users gives.
    It adds all the other default parameters and return the object.

    :param option: string. The type of learning rate desired. Supports:
                                                                        CosineDecay
                                                                        CosineDecayRestarts
    :param ParamDic: dictionary. If a parameter is in the dictionary then take that value, if not then use defaults.
    :return: A keras learning_rate_schedule object according to the request in the option parameter.
    """

    # Defaults parameters
    DefltInitLearning = 0.001  # 0.0001
    DefltSteps = 40
    DefltAlpha = 0.2
    DefltTMul = 0.2
    DefltMmul = 1.0
    defltNumPer = 0.5
    defltBeta = 0.001

    # Find parameters values.
    # If the values in the dic. take the value if not then use default

    ILR = ParamDic['initial_learning_rate'] if 'initial_learning_rate' in ParamDic else DefltInitLearning
    DS = ParamDic['decay_steps'] if 'decay_steps' in ParamDic else DefltSteps
    Alp = ParamDic['alpha'] if 'alpha' in ParamDic else DefltAlpha
    tMul = ParamDic['t_mul'] if 't_mul' in ParamDic else DefltTMul
    mMul = ParamDic['m_mul'] if 'm_mul' in ParamDic else DefltMmul
    NPer = ParamDic['num_periods'] if 'num_periods' in ParamDic else defltNumPer
    Beta = ParamDic['beta'] if 'beta' in ParamDic else defltBeta

    # pick the right decay according to option
    if option == 'CosineDecay':
        lr_decayed = CosineDecay(initial_learning_rate=ILR, decay_steps=DS, alpha=Alp, name='CosineDecay')
    elif option == 'CosineDecayRestarts':
        lr_decayed = CosineDecayRestarts(initial_learning_rate=ILR, first_decay_steps=DS, t_mul=tMul, m_mul=mMul,
                                         alpha=Alp, name='CosineDecayRestarts')
    else:
        print('Incorrect parameter:' + str(option))
        return

    return lr_decayed


def PickOptimizer(option='adam', ParamDic={}):
    """
    Build an optimizer object.

    It gets the type of optimizer needed and the parameters that the users gives.
    It adds all the other default parameters and return the object.

    Notice: SGD and adam support a learning rate decays the other don't

    :param option: string. The type of optimizer desired. Supports: adam
                                                                    Nadam
                                                                    Adamax
                                                                    SGD
    :param ParamDic: dictionary. If a parameter is in the dictionary then take that value, if not then use defaults.
    :return: A keras optimizer object as desired in the option parameter
    """

    # Defaults parameters
    DefltInitLearning = 0.001  # 0.0001
    DefltBeta1 = 0.9
    DefltBeta2 = 0.999
    DefltEpsil = 1e-07
    DefltAmsgrad = False
    DefltAMoment = 0.0
    Defltnesterov = False

    # Find parameters values
    # If the values in the dic. take the value if not then use default
    LR = ParamDic['learning_rate'] if 'learning_rate' in ParamDic else DefltInitLearning
    BT1 = ParamDic['beta_1'] if 'beta_1' in ParamDic else DefltBeta1
    BT2 = ParamDic['beta_2'] if 'beta_2' in ParamDic else DefltBeta2
    Eps = ParamDic['epsilon'] if 'epsilon' in ParamDic else DefltEpsil
    AMS = ParamDic['amsgrad'] if 'amsgrad' in ParamDic else DefltAmsgrad
    MMN = ParamDic['momentum'] if 'momentum' in ParamDic else DefltAMoment
    NES = ParamDic['nesterov'] if 'nesterov' in ParamDic else Defltnesterov

    if option == 'adam':
        opt = Adam(learning_rate=LR, beta_1=BT1, beta_2=BT2, epsilon=Eps, amsgrad=AMS, name='Adam')
    elif option == 'Nadam':  # Like adam but the momentum also get a direction
        opt = Nadam(learning_rate=LR, beta_1=BT1, beta_2=BT2, epsilon=Eps, name='Nadam')
    elif option == 'Adamax':
        # It is a variant of Adam based on the infinity norm.
        # Default parameters follow those provided in the paper.
        # Adamax is sometimes superior to adam, specially in models with embeddings.
        opt = Adamax(learning_rate=LR, beta_1=BT1, beta_2=BT2, epsilon=Eps, name='Adamax')
    elif option == 'SGD':
        opt = SGD(learning_rate=LR, momentum=MMN, nesterov=NES, name='SGD')
    else:
        print('Parameter option was not defined correctly')
        return

    return opt


def FindWeights(Generator, ExtraWeight={}):
    """
    Create a weights dictionary that used for balancing the data set classes. extra weight can also introduced

    The initial weights is calculated to balance the classes.
    Only then, if ExtraWeight dictionary is not empty we use the weights in the dictionary by multiply them by the
    already calculated weights.

    Example: Suppose 2 classes: [0,1] and unbalanced dataset of 100 photos for class 0 and 1000 photos for class 1.
    The default weights will be {0:10,1:1} that will balance the classes.
    Lets assume we want to give class 0 twice the weight of class 1.
    We will construct the following dictionary: {0:2,1:1}
    The result will be {0:20,1:1}

    :param Generator: keras ImageGenerator. The generator that contains the images that will enter the model
    :param ExtraWeight: dictionary. Contains the classes as keys and the extra weight as values
    :return: dictionary. Contains the classes as keys and the weights as values
    """
    NumInEachClass = Counter(Generator.labels.tolist())
    WeightsDic = {}
    for i in NumInEachClass.keys():
        WeightsDic[i] = (1 / NumInEachClass[i]) * Generator.n
    # In case we wants to add more weights
    if len(ExtraWeight) > 0:
        for i in NumInEachClass.keys():
            WeightsDic[i] = WeightsDic[i] * ExtraWeight[i]
    print('Num of pictures in each class:' + str(NumInEachClass))
    print('Weights:' + str(WeightsDic))

    return WeightsDic


def ShowEpochHistory(history):
    """
    Show 2 charts: One with loss function by epoch and the other is the accuracy chart
    :param history: The history that is received when doing model.fit
    :return:
    """
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    Accuracy_train = history.history['accuracy']
    Accuracy_val = history.history['val_accuracy']
    epochs = range(1, len(loss_train) + 1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    axes[0].plot(epochs, loss_train, 'g', label='Training loss', marker='*', markersize=10, markerfacecolor='blue',
                 markeredgecolor='#411a20', linewidth=3)
    axes[0].plot(epochs, loss_val, 'r', label='validation loss', marker='*', markersize=10, markerfacecolor='blue',
                 markeredgecolor='#411a20', linewidth=3)
    axes[1].plot(epochs, Accuracy_train, 'g', label='Training accuracy', marker='*', markersize=10,
                 markerfacecolor='blue', markeredgecolor='#411a20', linewidth=3)
    axes[1].plot(epochs, Accuracy_val, 'r', label='validation accuracy', marker='*', markersize=10,
                 markerfacecolor='blue', markeredgecolor='#411a20', linewidth=3)

    axes[0].set_title('Training and Validation loss', fontdict={'fontsize': 20, 'color': '#411a20'})
    axes[0].set_ylim([0, max(loss_train)])
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Training and Validation accuracy', fontdict={'fontsize': 20, 'color': '#411a20'})
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    MinTrain = min(Accuracy_train)
    MinVal = min(Accuracy_val)
    axes[1].set_ylim([min(MinTrain, MinVal), 1])
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].legend()

    plt.show()
