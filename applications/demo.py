#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__


import pathlib
from pathlib import Path
import sys
import json

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def get_files_of_type( source_folder, extensions ):

    valid_files = []

    for file in pathlib.Path(source_folder).glob('*.*'):

        ext = file.suffix
        if len(ext) > 1:
            ext = ext[1:]

        #print("ext " + ext )
        #print(str(file))
        if any(x in ext for x in extensions):
            valid_files.append( file )

    return valid_files

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def get_file_with_ext( file, ext ):
    return Path(file.parent / (file.stem + "." + ext))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def write_data_as_json( dat, file, pretty_print=False ):

    print_args = []
    if pretty_print:
        print_args = []

    f = open(file, "w")
    json_string = ""
    if pretty_print:
        json_string = json.dumps(dat, indent=4)
    else:
        json_string = json.dumps(dat)   
            
    #print(json_string)

    f.write( json_string )
    f.close()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def process_image_folder( source_folder ):

    print( "process_image_folder " + str(source_folder) )

    source_folder = Path(source_folder)

    image_extensions = ["png", "jpg", "tga", "bmp"]
    images_to_process = get_files_of_type( source_folder, image_extensions )

    output_folder = source_folder.parent / ( source_folder.stem + "_deep_out" )
    output_folder.mkdir(parents=True, exist_ok=True)

    currFileIndex = 1
    for file in images_to_process:

        image = cv2.imread(str(file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

        # create pose estimator
        image_size = image.shape

        pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

        # load model
        pose_estimator.initialise()

        # estimation
        pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

        pretty_print = True

        # Save 2D
        if len(pose_2d) > 0:
            pose_2d_json_out_file = Path(output_folder / (file.stem + "_pose2d.json"))
            write_data_as_json( pose_2d.tolist(), pose_2d_json_out_file, pretty_print )
            #print("Wrote " + str(pose_2d_json_out_file))
       else:
            print( "Empty pose_2d data" )

        # Save 3D
        if len(pose_3d) > 0:
            pose_3d_json_out_file = Path(output_folder / (file.stem + "_pose3d.json"))
            write_data_as_json( pose_3d.tolist(), pose_3d_json_out_file, pretty_print )
            #print("Wrote " + str(pose_3d_json_out_file))        
        else:
            print("Empty pose_3d data")

        # Save Visibility
        if len(visibility) > 0:
            visibility_json_out_file = Path(output_folder / (file.stem + "_visibility.json"))
            write_data_as_json( visibility.tolist(), visibility_json_out_file, pretty_print )
            #print("Wrote " + str(visibility_json_out_file))        
        else:
            print("Empty visibility data")

        # close model
        pose_estimator.close()

        currFileIndex = currFileIndex + 1
        print("Processed " + str(file) + "  " + str(currFileIndex) + "/" + str(len(images_to_process)) )

    # Show 2D and 3D poses
    #display_results(image, pose_2d, visibility, pose_3d)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if __name__ == '__main__':

    print( 'Number of arguments: ' + str(len(sys.argv)) + ' arguments.')
    print( 'Argument List: ' + str(sys.argv) )

    source_folder = sys.argv[1]

    sys.exit( process_image_folder(source_folder) )
