# -*- coding: utf-8 -*-
"""
This module deals with processing of video and images.

__print_progress() - Show the progress during a running of the video extraction in video_to_frames()

__extract_frames() - Used for the actual extraction of frames from video called from video_to_frames()

video_to_frames() - Takes a video, extract images, and save them

UseDirectory2CrtTrainFrames() - Extract a video to a specific path

CopyExtractFrames2TrainDir() - After the user chooses which image to leave in the directory then this function moves
                             the images to the relevant train path
preparePixelLibModel() - Read the model from the pixelLib repository.

Segment_a_Video() - Read a video and color all the pixels that are in the desired object (segmentation).

Segment_an_Image() - Read an image and color all the pixels that are in the desired object (segmentation).

ChangeBackg2Gray - Gets a video and then change all the pixels that are not the desired object to grayscale.

ChangeBackground - Gets a background image & a video. Change the pixels that aren't the object to the background image.

PrepareSSDMobile() - Read the SSD model from the repository

FindObjectInImage() - Gets an image and return the box around the desired object

CreateBoxAndLabelVideo() - Gets a video and return it rendered with a box around the desired object

CrtBoxAndLblImg() - Get an image ans the desired object and return the image with a box and a label of the class


"""
import numpy as np
import os
import sys
import cv2

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import copyfile
import urllib.request
from keras.layers import batchnormalization

import pixellib
import urllib
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg


def __print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar

    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """
    if total > 0:
        format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
        percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
        filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
        bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
        sys.stdout.flush()  # flush to stdout


def __extract_frames(video_path, frames_dir, overwrite=True, start=-1, end=-1, every=1, DifferntDirectory=True,
                     Name=''):
    """
    Extract frames from a video using OpenCVs VideoCapture

    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :param DifferntDirectory: If true then every video gets its own directory
    :param Name: If  DifferntDirectory=True then it adds the name to the file name
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            if DifferntDirectory:
                save_path = os.path.join(frames_dir, video_filename,
                                         "{:010d}.jpg".format(frame))  # create the save path
            else:
                OutFileName = Name + "{:010d}.jpg".format(frame)
                save_path = os.path.join(frames_dir, OutFileName)  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, image)  # save the extracted image
                saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=True, every=1, chunk_size=1000, DifferntDirectory=True, Name=''):
    """
    Extracts the frames from a video using multiprocessing

    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :param DifferntDirectory: If true then every video gets its own directory
    :param Name: If DifferntDirectory=True then it adds the name to the file name
    :return: path to the directory where the frames were saved, or None if fails
    """
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    if DifferntDirectory:
        # make directory to save frames, its a sub dir in the frames_dir with the video name
        os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    print(
        'Number 0f frames for ' + video_filename + ': ' + str(total) + '. Extracting every ' + str(every) + ' frames.')
    capture.release()  # release the capture straight away

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation")
        return None  # return None

    frame_chunks = [[i, i + chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    # make sure last chunk has correct end frame, also handles case chunk_size < total
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total - 1)

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

        futures = [
            executor.submit(__extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every, DifferntDirectory,
                            Name)
            for f in frame_chunks]  # submit the processes: __extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            __print_progress(i, len(frame_chunks) - 1, prefix=prefix_str, suffix='Complete')  # print it's progress
    if DifferntDirectory:
        return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames
    else:
        return frames_dir  # when done return the directory containing the frames


def UseDirectory2CrtTrainFrames(Path2InputDirectory, Path2Output, everyXFrames=10):
    """
    Extract images from video (every X frames) and put them in an output directory

    :param Path2InputDirectory: string. path to input directory
    :param Path2Output: string. path to output directory
    :param everyXFrames: int. the number of frames between each extracted image
    :return: Nothing
    """

    counter = 0
    for f in os.listdir(Path2InputDirectory):
        file_path = Path2InputDirectory + '/' + f
        counter = counter + 1
        VidName = 'Vid' + str(counter) + '_'
        if os.path.isfile(file_path):
            video_to_frames(video_path=file_path, frames_dir=Path2Output, overwrite=True, every=everyXFrames,
                            chunk_size=1000, DifferntDirectory=False, Name=VidName)


def CopyExtractFrames2TrainDir(Path2InputDirectory, Path2Output):
    """
    After extracting images from video and after the user filtered them, the images are copied to the output path
    :param Path2InputDirectory: string. input directory
    :param Path2Output: string. output directory
    :return: Nothing
    """
    counter = 0
    for f in os.listdir(Path2InputDirectory):
        inputName = os.path.join(Path2InputDirectory, f)
        OutputName = os.path.join(Path2Output, f)
        copyfile(inputName, OutputName)
        counter = counter + 1

    print('Files copied: ' + str(counter))


# The following function uses segmentation models that can detect classes in the image:
# Classes to find: person,bus,car,aeroplane, bicycle, ,motorbike,bird, boat, bottle, cat, chair, cow,
# dinningtable, dog, horse pottedplant, sheep, sofa, train, tv

def preparePixelLibModel(Modeltype='segmenation'):
    """
    This function retrieve the right model from the github repository.

    Currently it can get either a segmentation model or a model that changes the background.
    :param Modeltype: string. 'segmentation' for coloring object pixels.'Background' for changing the background.
    :return: Nothing
    """
    if Modeltype == 'segmenation':
        urllib.request.urlretrieve('https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5',
                                   'mask_rcnn_coco.h5')
    elif Modeltype == 'Background':
        urllib.request.urlretrieve(
            'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/xception_pascalvoc.pb',
            'xception_pascalvoc.pb')


def Segment_a_Video(PathInput, Pathoutput, AddBox=False):
    """
    This function takes a video and color the relevant object pixels

    :param PathInput: string. Input path
    :param Pathoutput: string. Output path
    :param AddBox: bool. If True then add boxes around the object
    :return: Nothing
    """
    cap = cv2.VideoCapture(PathInput)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('Pathoutput:' + str(Pathoutput))
    segment_video = instance_segmentation(infer_speed="fast")
    segment_video.load_model("mask_rcnn_coco.h5")
    segment_video.process_video(PathInput, show_bboxes=AddBox, frames_per_second=fps, output_video_name=Pathoutput)


def Segment_an_Image(PathInput, Pathoutput):
    """
    Color the pixel in an image where there are objects

    :param PathInput: string. Path input
    :param Pathoutput: string. Path output
    :return: Nothing
    """
    segment_image = instance_segmentation()
    segment_image.load_model("mask_rcnn_coco.h5")
    segment_image.segmentImage(PathInput, output_image_name=Pathoutput)


# works only on people
def ChangeBackg2Gray(PathInput, Pathoutput):
    """
    Get a video and grayscale the background leaving the objects in colors

    :param PathInput: string. Path input
    :param Pathoutput: string. Path output
    :return: Nothing
    """
    cap = cv2.VideoCapture(PathInput)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    change_bg = alter_bg(model_type="pb")
    change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    change_bg.gray_video(PathInput, frames_per_second=fps, output_video_name=Pathoutput, detect='person')


# works only on people
def ChangeBackground(BackgroundImage, PathInput, Pathoutput):
    """
    Change the background of a video to a specific image. The objects in the frames do not change

    :param BackgroundImage: np.ndarray . Image to set as the background
    :param PathInput: string. Path input
    :param Pathoutput: string. Path output
    :return: Nothing
    """
    cap = cv2.VideoCapture(PathInput)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    change_bg = alter_bg(model_type="pb")
    change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    change_bg.change_video_bg(PathInput, BackgroundImage, frames_per_second=fps, output_video_name=Pathoutput,
                              detect='person')


# The following few functions refers to: Object detection using SSD and MobileNet
# SSD = Single Shot Detectors

def PrepareSSDMobile():
    """
    Get the object detection model from repository
    :return: cv2.dnn_Net object that contains the model for object detection
    """
    # load our serialized model from disk
    print("[INFO] loading model...")
    urllib.request.urlretrieve(
        'https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy'
        '.caffemodel?raw=true',
        'MobileNetSSD_deploy.caffemodel')
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/nikmart/pi-object-detection/master/MobileNetSSD_deploy.prototxt.txt',
        'MobileNetSSD_deploy.prototxt.txt')
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    return net


def FindObjectInImage(Img, What2Find, net):
    """
    Return a box over the desired object in an image

    :param Img: np.ndarray. An input image
    :param What2Find: List with usually one string. This is the type of object to look for in the image
    :param net: cv2.dnn_Net. The model used for running the object detection
    :return: Tuple that contains the box parameters around the object, the preferred location of the y acording to how
             close the y to the end of the image. If no relevant object found then return ([-1], 0)
    """
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    ChosenClass = [CLASSES.index(x) for x in What2Find]

    # ChosenClass = CLASSES.index(What2Find) #find the index of the relevant class in list CLASSES
    # Read and preprocess
    image = Img
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # extract the index of the class label from the `detections`

        idx = int(detections[0, 0, i, 1])
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence and make sure we have the right class

        if (confidence > 0.5) and (idx in ChosenClass):

            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            yForText = startY - 15 if startY - 15 > 15 else startY + 15
            return box.astype("int"), yForText
        else:
            return [-1], 0


def CreateBoxAndLabelVideo(net, VideoPath, videoName, VideoOutputPath, Label, MainClass, Fontcolor=(204, 0, 0),
                           BoxColor=(0, 0, 255)):
    """
    Create a video with Box and label over the subject.

    :param net: cv2.dnn_Net. The model used for running the object detection.
    :param VideoPath: string. The path to the video.
    :param videoName: string. File name.
    :param VideoOutputPath: string. Output directory name.
    :param Label: string. What to use as text over the detected object.
    :param MainClass: List with usually one string. This is the type of object to look for in the image.
    :param Fontcolor: tuple. This is the RGB of the font color.
    :param BoxColor: tuple. This is the RGB for the box color.
    :return: Nothing.
    """
    # read the video
    cap = cv2.VideoCapture(VideoPath)
    # Get the parameters of the input video
    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # Create an ouput video usig parameters from the input video
    outFileName = Label + '_' + videoName
    # Create the output video
    out = cv2.VideoWriter(VideoOutputPath + '/' + outFileName, fourcc, fps, (width, height))
    # out = cv2.VideoWriter('output.mp4',fourcc, fps, (width,height))
    FrameNum = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    FontColor = Fontcolor
    prevX = 0
    prevY = 0
    threshold = 0.1
    while True:

        # Capture frames in the video
        ret, frame = cap.read()
        FrameNum = FrameNum + 1

        if frame is None:
            break
        # Search for object in frame
        (OutBox, yForText) = FindObjectInImage(frame, MainClass, net)
        # if the OutBox is contains more then 1 value then it means that an object was found
        if len(OutBox) > 1:
            # Calculate the location of the box and label
            (startX, startY, endX, endY) = OutBox

            x = int((startX + endX) / 2)
            y = int((endY - startY) * 0.2)

            # To avoid moving the frame and the label every frame we use a threshold that if the change in x or
            # the change in y is greater than it then we update the location of the label and box

            diffX = abs(x - prevX) / (endX - startX)
            diffy = abs(y - prevY) / (endY - startY)

            if diffX < threshold:
                currX = prevX
            else:
                currX = x
            if diffy < threshold:
                currY = prevY
            else:
                currY = y

            # write on the image the rectangle and the text
            cv2.rectangle(frame, (startX, startY), (endX, endY), BoxColor, thickness=3)
            cv2.putText(frame, Label, (currX, currY), font, 2, FontColor, 3)
            out.write(frame)
            prevY = currY
            prevX = currX

    # release the cap object
    cap.release()
    out.release()


def CrtBoxAndLblImg(image, Class2Find, net, Label, Fontcolor=(204, 0, 0), BoxColor=(0, 0, 255), FontScale=3,
                    Fnthickness=3):
    """
    Create a box and a label over the desired object

    :param image: np.ndarray. An input image
    :param Class2Find: list. What class we are looking for
    :param net: cv2.dnn_Net. The model used for running the object detection.
    :param Label: string. The label to put over the object
    :param Fontcolor: tuple RGB. Font color
    :param BoxColor: tuple RGB. Box color
    :param FontScale: int. How big should be the font
    :param Fnthickness: int. Font thickness
    :return: np.ndarray. An image with the box and the label
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (OutBox, yForText) = FindObjectInImage(image, Class2Find, net)
    if len(OutBox) > 1:
        # Calculate the location of the box and label
        (startX, startY, endX, endY) = OutBox
        fc = Fontcolor
        # find the location to put the box and label.
        x = int((startX + endX) / 2)
        y = int((endY - startY) * 0.2)
        cv2.rectangle(image, (startX, startY), (endX, endY), BoxColor, thickness=3)
        cv2.putText(image, Label, (x, y), font, FontScale, fc, Fnthickness)
        return image
