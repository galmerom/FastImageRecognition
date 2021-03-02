# -*- coding: utf-8 -*-
"""
This module creates the Paths dictionary and the list of desired imagnet classes

CreateProjectDirectories(): Gets the main path of the project and creates a dictionary of the needed directory
                          according to hierarchy.
createDirectories(): Get a dictionary with the desired paths and hierarchy and open the directories

CleanDirectoryFromFiles(): Delete a directory from files

GetImagenetClasses(): Gets the imagenet files from github

LstRelevImagnetClasses(): Creates a list of the relevant files

"""

import os
import urllib
import json


def CreateProjectDirectories(MainPath, ListOfClasses, MainClass):
    """
    Creates a dictionary of paths from the input main path that are needed in the project

    :param MainPath: string. The main path
    :param ListOfClasses: List of desired classes in the project. Every class will get its own directory in the output
    :param MainClass: string. The main class (also used for output directory)
    :return: A dictionary with the desired paths
    """
    x = {}

    x[MainPath] = ['train_dataset', 'test_dataset', 'save model and weights', 'augmented pictures', 'Video input',
                   'Video extracted frames', 'Output', 'Video for training data', 'temp dir', 'Batch input images']
    x[MainPath + '/Video for training data'] = ['VideoTrainFrames']
    x[MainPath + '/train_dataset'] = ListOfClasses
    x[MainPath + '/test_dataset'] = ListOfClasses
    outList = [MainClass[0] + ' without specific classes']
    outList.extend(ListOfClasses)
    outList.append('Nothing deteced')
    x[MainPath + '/Output'] = outList
    x[MainPath + '/save model and weights'] = ['Saved model', 'Saved weights']
    return x


def createDirectories(dicDirectry):
    """
    Creates all the needed directories for the project.

    Returns a dictionary that holds the path for every item in the input dictionary
    Currently can only be used for main directory and only 2 level deep

    :param dicDirectry: Dictionary that holds the paths needed as keys and the name of the directory as values
    :return: A dictionary that holds the names of the directory as keys and the paths as values
    """
    MainDir = list(dicDirectry.keys())[0]
    paths = {'Main': MainDir}
    for item in dicDirectry.keys():
        print(item)
        for dirc in dicDirectry[item]:
            newPath = item + '/' + dirc
            print(newPath)
            if not os.path.exists(newPath):
                os.mkdir(newPath)
            if dirc in paths.keys():
                # in case we try to open a path with the same name then we will use the parent directory as a prefix
                Updirc = item.split('/')[-1]
                paths[Updirc + '_' + dirc] = newPath
            else:
                paths[dirc] = newPath
    return paths


def CleanDirectoryFromFiles(Path):
    """
    Cleans a directory from files
    :param Path: string. The directory path
    :return: Nothing
    """
    Currdir = Path
    for f in os.listdir(Path):
        if os.path.isfile(Path + '/' + f):
            os.remove(os.path.join(Currdir, f))


# Imagnet classes


def GetImagenetClasses():
    """
    Import the imagnet classes from  github.
    :return: Dictionary with index as keys and and a list of: class name and classes description
    """
    url = 'https://raw.githubusercontent.com/marcotcr/lime/master/doc/notebooks/data/imagenet_class_index.json'
    response = urllib.request.urlopen(url)
    ImgnetDic = json.loads(response.read())
    return ImgnetDic


def LstRelevImagnetClasses(option, ClassIndxList):
    """
    Gets an input statement to what imagenet classes to retrieve and returns only the classes that were asked



    :param option: int. If option = 1 then the input statement will be a list of specific classes numbers in a list
                                    (the input are the indexes in the  dictionary retrieved from GetImagenetClasses() )
                        If option = 2 gets a list of tuples. Each represents "from" class number "to" class number
                                    (the input are the indexes in the  dictionary retrieved from GetImagenetClasses() )
    :param ClassIndxList: A list in the format defined in the option parameter
    :return: A list of the desired classes (for example:'n07717410')
    """
    imagnetDic = GetImagenetClasses()
    ClassList = []
    if option == 1:
        for item in ClassIndxList:
            ClassList.append(imagnetDic[str(item)][0])
    elif option == 2:
        for tpl in ClassIndxList:
            for item in imagnetDic.keys():
                if tpl[0] <= int(item) <= tpl[1]:
                    ClassList.append(imagnetDic[item][0])

    return ClassList
