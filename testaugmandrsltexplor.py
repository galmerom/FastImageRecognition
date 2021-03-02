# -*- coding: utf-8 -*-
"""

This module is used for testing the model and explore the results.

TestAugmenByBatch() - extract the image and send it to TTA for prediction. Then send labels and prediction to
ExplorePredResults

PredictInBatch() - Do the same as TestAugmenByBatch without TTA.

ExplorePredResults() - call plotCM and  show mispredict photos with label ver. prediction

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import expand_dims
import cv2

from sklearn.metrics import confusion_matrix, classification_report


# make a prediction using test-time augmentation
def _tta_prediction(ImageGenerator, model, image, nExamples):
    """
    Perform test time augmentation.

    Create the test time augmentation per received image.
    Creates X augmented images.
    Then max the sum of all prediction squared to get the more relevant class.

    Args:
    :param ImageGenerator: an augmentation generator instance.
    :param model: A Keras model.
    :param image: One image.
    :param nExamples: An integer number for the number of augmentation images to create.
    :return: An integer with best predicted class.
    """
    # convert image into dataset
    samples = expand_dims(image, 0)
    # prepare iterator
    BatchIterator = ImageGenerator.flow(samples, batch_size=nExamples, seed=42)
    # make predictions for each augmented image
    yPred = model.predict(BatchIterator, verbose=0)
    # sum across predictions
    summed = np.sum(yPred ** 2, axis=0)
    # argmax across classes
    return np.argmax(summed)


# Make a test time augmentation prediction
def TestAugmenByBatch(ImageGener, TTAugmentation, model, NumberOfSteps, NumOfImgAugm=5, diffImgTool=False):
    """
    Use this function for creating test time augmentation and explore the output

    It unpack a batch of images from the image generator. Then take one image at a time and send it to the TTA.
    It collects all the labels and the prediction of all images and send them to ExplorePredResults to allow the user
    to explore the output
    :param ImageGener: An ImageDataGenerator instance used for reading the images
    :param TTAugmentation: An ImageDataGenerator instance used for creating the augmented images
    :param model:  A Keras model
    :param NumberOfSteps: An integer that states how many time to do the extraction from the image generator
    :param NumOfImgAugm: int of the number of images to generate for any input image
    :param diffImgTool: If True don't show different between predicted and True labels
    :return: a tuple: (ProblemPic, y_pred, y_trueList). ProblemPic = list of tuples that contains
     predicted images and their true labels with predicted labels
    """
    y_pred = []  # Holds all the predictions from all batches
    y_trueList = []  # Holds the true labels from all batches
    ProblemPic = []  # Counts the batch number
    StepCounter = 0

    # For each batch get the images and the labels
    for imgBatch, label in ImageGener:
        # Send every single image to TTA
        for i in range(0, len(imgBatch)):
            pred = _tta_prediction(TTAugmentation, model, imgBatch[i], NumOfImgAugm)
            labels = np.argmax(label[i]).tolist()
            y_pred.append(pred)
            y_trueList.append(labels)

            # in case the prediction and the label are not the same append them to a list of tuples
            if pred != labels:
                ProblemPic.append((pred, labels, imgBatch[i]))

        # Go to next batch unless reached the number of planned steps
        StepCounter = StepCounter + 1
        if StepCounter > NumberOfSteps:
            break

    ExplorePredResults(ProblemPic, y_pred, y_trueList, ImageGener.class_indices, diffImgTool)
    return ProblemPic, y_pred, y_trueList


def PredictInBatch(ImageGener, model, NumberOfSteps, diffImgTool=False):
    """
    Use this function to predict and compare against the labels WITHOUT TTA
    :param ImageGener: An image generator instance
    :param model: A Keras model
    :param NumberOfSteps: How many time to do the extraction from the image generator
    :param diffImgTool: If True don't show different between predicted and True labels
    :return: a tuple: (ProblemPic, y_pred, y_trueList). ProblemPic = list of tuples that contains
     predicted images and their true labels with predicted labels
    """
    y_pred = []  # Holds all the predictions from all batches
    y_trueList = []  # Holds the true labels from all batches
    ProblemPic = []  # Counts the batch number
    StepCounter = 0

    for img, label in ImageGener:
        pred = model.predict(img)
        pred = np.argmax(pred, axis=1).tolist()
        labels = np.argmax(label, axis=1).tolist()

        # Creates 2 lists of prediction and y_true for confusion matrix. Holds data for all batches
        y_pred.extend(pred)
        y_trueList.extend(labels)

        ZipList = list(zip(pred, labels))

        axCounter = 0
        for item in ZipList:
            if item[0] != item[1]:
                ProblemPic.append((item[0], item[1], img[axCounter]))
            axCounter = axCounter + 1

        # Go to next batch unless reached the number of planned steps
        StepCounter = StepCounter + 1
        if StepCounter > NumberOfSteps:
            break

    ExplorePredResults(ProblemPic, y_pred, y_trueList, ImageGener.class_indices, diffImgTool)
    return ProblemPic, y_pred, y_trueList


def ExplorePredResults(ProblemPic, y_pred, y_trueList, class_indices, diffImgTool=False):
    """
    Create a confusion matrix, classification report and a tool that shows the unpredicted images

    :param ProblemPic: A list of tuples that each contains: (predicted label, true label, image)
    :param y_pred: A list of all the predicted values
    :param y_trueList: A list of all the true values
    :param class_indices: An image generator argument contains: dictionary with classes and their real name
    :param diffImgTool: If True don't show different between predicted and True labels
    :return: Nothing
    """
    # Creates the confusion matrix and the classification report
    plotCM(y_pred, y_trueList, class_indices)

    # preparing the size of plt.figure
    NumOfPic = len(ProblemPic)
    rows = int(NumOfPic / 3) + 1
    LengthOfFig = int(rows * 7.5)
    fig = plt.figure(figsize=(20, LengthOfFig))

    if diffImgTool:
        return
    # plot image tool. Images with the predicted value and the true value

    for i in range(0, len(ProblemPic)):
        ax = fig.add_subplot(rows, 3, i + 1, xticks=[], yticks=[])
        predict, y_true, img = ProblemPic[i]

        # decode the preprocess
        img = img + [103.939, 116.779, 123.68]  # Add the "original" mean
        img = img[..., ::-1]  # go back from BRG to RGB
        img = img.astype('int')  # The values must be integers

        # Decode the classes names
        classPredict = list(class_indices.keys())[ProblemPic[i][0]]
        classActual = list(class_indices.keys())[ProblemPic[i][1]]
        txt = ' \n\n' + 'Predict:' + classPredict + '\nActual: ' + classActual + '\n'

        ax.imshow(img, cmap=plt.cm.bone, origin='upper')
        ax.text(0, 0, txt, fontsize=20)


def plotCM(y_pred, y_true, classes,
           normalize=False,
           title=None,
           cmap=plt.cm.Blues,
           precisionVal=2,
           titleSize=15,
           fig_size=(7, 5),
           InFontSize=15,
           LabelSize=15,
           ClassReport=True):
    """
    This function prints and plots the confusion matrix.


    :param y_pred: List of predicted labels
    :param y_true: List of true labels
    :param classes: A dictionary with the name of the classes and their values
    :param normalize: bool. True means every line of the confusion matrix show the % of each square from the whole raw
    :param title: string Chart title
    :param cmap: color map
    :param precisionVal: integer number of digits after the dot (example: 0.00 = 2)
    :param titleSize: int. title font size
    :param fig_size: tuple Figure size (width,height)
    :param InFontSize: int. The font of the values inside the table
    :param LabelSize: int. Label font size
    :param ClassReport: bool. If true add a classification report at the bottom
    :return: Nothing
    """
    # Set a default title in case none given
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title
           )
    ax.xaxis.set_tick_params(labelsize=LabelSize)
    ax.yaxis.set_tick_params(labelsize=LabelSize)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.' + str(precisionVal) + 'f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontdict={'fontsize': InFontSize})
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true)) - 0.5)
    plt.ylim(len(np.unique(y_true)) - 0.5, -0.5)
    plt.xlabel(xlabel='Predicted label', fontdict={'fontsize': 15, 'color': '#411a20'})
    plt.ylabel(ylabel='True label', fontdict={'fontsize': 15, 'color': '#411a20'})
    plt.title(title + '\n', fontdict={'fontsize': titleSize, 'color': '#411a20'})
    plt.show()
    if ClassReport:
        print('\n\nClassification_report\n*********************\n')
        print(classification_report(y_true=y_true,
                                    y_pred=y_pred))
