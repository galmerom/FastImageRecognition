# -*- coding: utf-8 -*-
"""
This module help find an object in a video or an image

FindObjInVideoDirectory(): Get a directory with videos ans filter to another directory the ones that contains the
                         desired object. It also allows  the use of object detection or segmentation over the video
FindObjIn1Video(): Get one video and find out if the desired object is there.It also allows the user to use object
                   detection or segmentation over the video
__CrtOutputVideo(): Change a video to contains boxes over drawn around the desired object
               or render every pixel of the desired object
FindObjInbatchImgs(): Gets a directory with images and sort them to contains/don't contains the desired object

FindObjectIn1Pic(): Get one image and sort it to contains/don't contains the desired object

__DetectClass(): Get an image and find out if a specific class is there or not

showImages(): Show 2 images one that contains a box around the desired object and next to it, the original image.

__IsImgntClassInImg(): Check if the class detected in the image are in the predefined classes list.
                     Works on the general model only (referred to in the module as: First model)

__FindClassInSpesificModel(): Check if the specific classes in the specific model is in the image

__preProcessImg(): Preprocess the image to fit the desired model

__CleanDirectory(): Removes all files from the directory

__UnprepareImg(): Change the image from a BRG to RGB format
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys

from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications import imagenet_utils
from keras.applications import imagenet_utils

from shutil import copyfile
from collections import Counter
import detectinvideo as videct


def FindObjInVideoDirectory(DirecPath, Firstmodel, model, imgntClasses, ClassesInModel, ImprtntClss, MainClass, paths,
                            TypeOfVideo='SSD'):
    """
    Get a directory of videos and sort them to the ones that contain the desired object and to the rest.

    It also allows object detection boxes or pixel segmentation rendered in the video

    :param DirecPath: string. Directory path
    :param Firstmodel: Image classification Keras model (for example: resnet50 from keras)
    :param model: Keras model. The specific model used to find a specific object. Model must be a model using keras
    :param imgntClasses: A list of classes containing the classes to look for in the first model
    :param ClassesInModel: A list of classes to look for in the specific model
    :param ImprtntClss: string. The "important class". The class that we are looking for
    :param MainClass: A list with one string.The main class of the object used for object detection must be
                      one of the following:
                      person, bus, car, aeroplane, bicycle, ,motorbike,bird, boat, bottle, cat, chair, cow,
                      dinningtable, dog, horse, pottedplant, sheep, sofa, train, tv
    :param paths: A dictionary that contains the names of the paths as keys and the actual paths as values
    :param TypeOfVideo: string. put 'SSD' to put boxes over the object.'Segmentation' to get the object pixels colored
    :return: Nothing
    """
    # Cleans the directory
    for lbls in ClassesInModel:
        __CleanDirectory(paths['Output_' + lbls])

    # If SSD type video was chosen import the model and run FindObjIn1Video for every file
    if TypeOfVideo == 'SSD':
        # Gets the model
        net = videct.PrepareSSDMobile()
        for f in os.listdir(DirecPath):
            file_path = DirecPath + '/' + f
            FindObjIn1Video(file_path, Firstmodel, model, imgntClasses, ClassesInModel, ImprtntClss, MainClass, paths,
                            TypeOfVideo='SSD', SSDNet=net)
    # If Segmentation type video was chosen import the model and run FindObjIn1Video for every file
    if TypeOfVideo == 'Segmentation':
        videct.preparePixelLibModel()
        print('Number of videos in input: ' + str(len(os.listdir(DirecPath))))
        print('DirecPath:'+DirecPath)
        print(os.listdir(DirecPath))
        for f in os.listdir(DirecPath):
            file_path = DirecPath + '/' + f
            FindObjIn1Video(file_path, Firstmodel, model, imgntClasses, ClassesInModel, ImprtntClss, MainClass, paths,
                            TypeOfVideo='Segmentation')


def FindObjIn1Video(InpFilePath, Firstmodel, model, imgntClasses, ClassesInModel, ImprtntClss, MainClass, paths,
                    TypeOfVideo='SSD', SSDNet=None):
    """
    Find if an object is in the video or not. If desired add object detection or pixel rendering of the object

    :param InpFilePath: String. The path to the image file
    :param Firstmodel: Image classification Keras model (for example: resnet50 from keras)
    :param model: Keras model. The specific model used to find a specific object. Model must be a model using keras
    :param imgntClasses: A list of classes contains the classes to look for in the first model
    :param ClassesInModel: A list of classes to look for in the specific model
    :param ImprtntClss: string. The "important class". The class that we are looking for
    :param MainClass: A list with one string.The main class of the object used for object detection must be
                      one of the following:
                      person, bus, car, aeroplane, bicycle, ,motorbike,bird, boat, bottle, cat, chair, cow,
                      dinningtable, dog, horse, pottedplant, sheep, sofa, train, tv
    :param paths: A dictionary that contains the names of the paths as keys and the actual paths as values
    :param TypeOfVideo: string. put 'SSD' to put boxes over the object.'Segmentation' to get the object pixels colored
    :param SSDNet: cv2.dnn_Net: If object detection needed then this parameter holds the model that holds the
                   SSD network. The retrievement of the model is done only once at the directory level.
    :return: Nothing
    """
    ImpFlag = False
    filename = os.path.basename(InpFilePath)
    # Clean the frames  directory
    ExtrcPath = paths['Video extracted frames']
    __CleanDirectory(ExtrcPath)

    # extract frames from video and put them in the frames directory
    videct.video_to_frames(InpFilePath, ExtrcPath, every=10, DifferntDirectory=False)
    # Go over all extracted frames and copy the frames to the relevant directory
    for f in os.listdir(ExtrcPath):
        file_path = ExtrcPath + '/' + f
        print(file_path)
        # Checks if the desired specific object is in the image. If nothing found return 'Found Nothing'
        Ans = __DetectClass(file_path, Firstmodel, model, imgntClasses, ClassesInModel)
        print(Ans)
        if Ans != 'Found Nothing':
            print(paths['Output_' + Ans] + '/' + filename)
            copyfile(file_path, paths['Output_' + Ans] + '/' + filename)
            if Ans == ImprtntClss:
                ImpFlag = True
                break

    # If the desired class was found, check the  TypeOfVideo parameter and send it to __CrtOupVideo or just copy the
    # file to the relevant directory
    if ImpFlag:
        if TypeOfVideo == 'SSD':
            __CrtOutputVideo(TypeOfVideo, InpFilePath, filename, paths, ImprtntClss, MainClass, SSDNet=SSDNet)
        elif TypeOfVideo == 'Segmentation':
            __CrtOutputVideo(TypeOfVideo, InpFilePath, filename, paths, ImprtntClss, MainClass)
        else:
            copyfile(InpFilePath, paths['Output_' + ImprtntClss] + '/' + filename)


def __CrtOutputVideo(VidType, VideoPath, FileName, paths, ImprtntClss, MainClass, SSDNet=None):
    """
    Create the output video: an image detect or segmentation video (color every pixel of the object)
    :param VidType: string. The desired action. Can get: 'SSD' or 'Segmentation'.
    :param VideoPath: string. Path to the location of the video.
    :param FileName: string. The video file name.
    :param paths: A dictionary that contains the names of the paths as keys and the actual paths as values.
    :param ImprtntClss: string. The "important class". The class that we are looking for.
    :param MainClass: A list with one string.The main class of the object used for object detection must be
                      one of the following:
                      person, bus, car, aeroplane, bicycle, ,motorbike,bird, boat, bottle, cat, chair, cow,
                      dinningtable, dog, horse, pottedplant, sheep, sofa, train, tv
    :param SSDNet: cv2.dnn_Net: If object detection needed then this parameter holds the model that holds the
                   SSD network. The retrievement of the model is done only once at the directory level.
    :return: Nothing
    """
    OutPath = paths['Output_' + ImprtntClss]
    if VidType == 'SSD':
        videct.CreateBoxAndLabelVideo(SSDNet, VideoPath, FileName, OutPath, ImprtntClss, MainClass,
                                      Fontcolor=(204, 0, 0), BoxColor=(0, 0, 255))
    if VidType == 'Segmentation':
        print('Segmentation')
        filename = os.path.basename(VideoPath)
        videct.Segment_a_Video(VideoPath, OutPath+'/'+filename, AddBox=False)


def FindObjInbatchImgs(Dir_path, Firstmodel, model, paths, MainClass, imgntClasses, ClassesInModel, fontScale=3,
                       FontThick=3, boxes=True, ColorObj=False):
    """
    Find a specific object in images that are in a directory. Copy the images to a sorted directories
    :param Dir_path: string. Directory path
    :param Firstmodel: Image classification Keras model (for example: resnet50 from keras)
    :param model: Keras model. The specific model used to find a specific object. Model must be a model using keras
    :param paths: A dictionary that contains the names of the paths as keys and the actual paths as values
    :param MainClass: A list with one string.The main class of the object used for object detection must be
                      one of the following:
                      person, bus, car, aeroplane, bicycle, ,motorbike,bird, boat, bottle, cat, chair, cow,
                      dinningtable, dog, horse, pottedplant, sheep, sofa, train, tv
    :param imgntClasses: A list of classes contains the classes to look for in the first model
    :param ClassesInModel: A list of classes to look for in the specific model
    :param fontScale: int. The font scale to use when writing on the image
    :param FontThick: int. The font thickness to use when writing on the image
    :param boxes: bool. If True then add box over the picture
    :param ColorObj: bool. If True then color the pixel of the image that contains the desired object
    :return: Nothing
  """
    # Cleans the output directory
    for lbl in ClassesInModel:
        __CleanDirectory(paths['Output_' + lbl])

    # Run for every image found in the directory
    for f in os.listdir(Dir_path):
        file_path = Dir_path + '/' + f
        FindObjectIn1Pic(file_path, Firstmodel, model, paths, MainClass, imgntClasses, ClassesInModel,
                         fontScale=fontScale, FontThick=FontThick, boxes=boxes, ColorObj=ColorObj, RtrTyp=False)


def FindObjectIn1Pic(img_path, Firstmodel, model, paths, MainClass, imgntClasses, ClassesInModel, fontScale=3,
                     FontThick=3, boxes=True, ColorObj=False, RtrTyp=True):
    """
    Find a specific object in an image

    :param img_path: string. Image path
    :param Firstmodel: Image classification Keras model (for example: resnet50 from keras)
    :param model: Keras model. The specific model used to find a specific object. Model must be a model using keras
    :param paths: A dictionary that contains the names of the paths as keys and the actual paths as values
    :param MainClass: A list with one string.The main class of the object used for object detection must be
                      one of the following:
                      person, bus, car, aeroplane, bicycle, ,motorbike,bird, boat, bottle, cat, chair, cow,
                      dinningtable, dog, horse, pottedplant, sheep, sofa, train, tv
    :param imgntClasses: A list of classes contains the classes to look for in the first model
    :param ClassesInModel: A list of classes to look for in the specific model
    :param fontScale: int. The font scale to use when writing on the image
    :param FontThick: int. The font thickness to use when writing on the image
    :param boxes: bool. If True then add box over the picture
    :param ColorObj: bool. If True then color the pixel of the image that contains the desired object
    :param RtrTyp: bool. If True then show the images (The original image next to the image with the detection box)
                   If Not true then copy the image with object detection box to the output directory
    :return: Nothing
    """
    # Looks for the desired object in the image. If a class is find then Ans will get the class
    # if not it will get 'Found Nothing'
    Ans = __DetectClass(img_path, Firstmodel, model, imgntClasses, ClassesInModel)
    print(Ans)
    if Ans == 'Found Nothing':
        return
    PredClass = Ans
    OutImg = None
    if boxes:
        # Gets the SSD model
        net = videct.PrepareSSDMobile()
        CV_image = cv2.imread(img_path)
        # Add a box to the image where the object is
        OutImg = videct.CrtBoxAndLblImg(CV_image, [MainClass], net, PredClass, FontScale=fontScale,
                                        Fnthickness=FontThick)
    if ColorObj:
        CheckFile = 'mask_rcnn_coco.h5'
        if CheckFile not in sys.modules:
            videct.preparePixelLibModel(Modeltype='segmenation')
        filename = os.path.basename(img_path)
        Dir = os.path.dirname(img_path)
        # Color the image and save it to the directory it got it from with prefix SEG_
        videct.Segment_an_Image(img_path, Dir + 'SEG_' + filename)
        # Read the image and remove the SEG_ file
        OutImg = cv2.imread(Dir + 'SEG_' + filename)
        os.remove(Dir + 'SEG_' + filename)

    if OutImg is not None:
        if RtrTyp:
            showImages(img_path, OutImg)
        else:
            outpath = paths['Output_' + PredClass] + '/' + os.path.basename(img_path)
            cv2.imwrite(outpath, OutImg)


def __DetectClass(img_path, Firstmodel, model, imgntClasses, ClassesInModel):
    """
    Detect if a specific class is in an image

    This is a 2 steps process. First it looks in the first model if it founds the relevant class there it send it to the
    specific model and look if it is there.

    :param img_path: Image path
    :param Firstmodel: Image classification Keras model (for example: resnet50 from keras)
    :param model: Keras model. The specific model used to find a specific object. Model must be a model using keras
    :param imgntClasses: A list of classes contains the classes to look for in the first model
    :param ClassesInModel: A list of classes to look for in the specific model
    :return: returns a tuple with 2 elements.(string of class name, int class index)
    """

    CurrImg = __preProcessImg(img_path)
    if not __IsImgntClassInImg(Firstmodel, CurrImg, imgntClasses):
        return 'Found Nothing'
    PredClass = __FindClassInSpesificModel(model, CurrImg, ClassesInModel)[0]
    return PredClass


def showImages(OriginalImagePath, img, RGBProb=True):
    """
    Show images one next to the other.

    Used for showing original image against a changed image

    :param OriginalImagePath: string. Original image path
    :param img: np.ndarray. An image in BRG format
    :param RGBProb: bool. If True then it transform the image to RGB
    :return: Nothing
    """
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    orig = cv2.imread(OriginalImagePath)
    if RGBProb:
        # Change the format of the image to RGB
        orig = __UnprepareImg(orig)
    ax.imshow(orig, cmap=plt.cm.bone, origin='upper')
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    img = __UnprepareImg(img)
    ax.imshow(img, cmap=plt.cm.bone, origin='upper')


def __IsImgntClassInImg(Firstmodel, img, ClassList):
    """
    Gets a model, an image and a class list. Check if the image contains an object that is in the class list
    :param Firstmodel: Image classification Keras model (for example: resnet50 from keras)
    :param img: np.ndarray. An image after preprocessing
    :param ClassList: List. An imagenet  classes list that we want the model to find in the image
    :return: bool. True if the object was found else False
    """
    preds = Firstmodel.predict(img)
    pred = imagenet_utils.decode_predictions(preds)[0][0][0]
    # check if the predicted class is in the True classes (ClassList)
    if pred in ClassList:
        return True
    else:
        return False


def __FindClassInSpesificModel(model, img, ClassesInModel):
    """
    Find an object in the specific model. Return True or False if found

    :param model: Keras model. The specific model used to find a specific object. Model must be a model using keras
    :param img: np.ndarray. An image after preprocessing
    :param ClassesInModel: A dictionary of classes for the specific objects
    :return: A tuple of: The class found as a string and the index of the class
    """
    y_pred = model.predict(img)
    y_pred = int(np.argmax(y_pred, axis=1))
    classMeans = ClassesInModel[y_pred]
    return classMeans, y_pred


def __preProcessImg(img_path):
    """
    Gets an image and return a preprocessed image
    :param img_path: string path to image
    :return: np.ndarray. Image (After preprocessing)
    """
    x = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    return x


def __CleanDirectory(DirecPath):
    """
    Get a directory path and remove all the files it contains
    :param DirecPath: string. Path to the directory
    :return: Nothing
    """
    Currdir = DirecPath
    for f in os.listdir(DirecPath):
        if os.path.isfile(DirecPath + '/' + f):
            os.remove(os.path.join(Currdir, f))


def __UnprepareImg(img):
    """
    Change image from BRG to RGB format

    :param img: np.ndarray. Image in BRG format
    :return: np.ndarray. Image in RGB format
    """
    img = img[..., ::-1]  # go back from BRG to RGB
    img = img.astype('int')  # The values must be integers
    return img
