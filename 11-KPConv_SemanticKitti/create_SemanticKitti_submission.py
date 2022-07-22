#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to create a semanticKitti submission
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from utils.tester import ModelTester
from models.KPCNN_model import KernelPointCNN
from models.KPFCNN_model import KernelPointFCNN

# Datasets
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.ShapeNetPart import ShapeNetPartDataset
from datasets.S3DIS import S3DISDataset
from datasets.Scannet import ScannetDataset
from datasets.NPM3D import NPM3DDataset
from datasets.Semantic3D import Semantic3DDataset
from datasets.SemanticKitti import SemanticKittiDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#



# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Choose the model to test
    ##########################

    #
    #       > 'Log_YYYY-MM-DD_HH-MM-SS': Name of the log folder
    #

    chosen_log = 'Log_2019-11-07_19-28-12'

    log_path = os.path.join('results', chosen_log)
    test_path = os.path.join('test', chosen_log)

    # Check if the log exist and has been tested
    if os.path.exists(log_path):

        if os.path.exists(test_path):
            print('Starting the convertion of', chosen_log)
        else:
            print(chosen_log, 'exists but has not been tested')
    else:
        print(chosen_log, 'does not exists')

    ##########################
    # Choose the model to test
    ##########################

    # Load model parameters
    config = Config()
    config.load(log_path)

    # Initiate dataset configuration
    dataset = SemanticKittiDataset(config.input_threads, n_frames=config.n_frames)

    submission_folder = os.path.join(test_path, 'sequences')
    if not os.path.exists(submission_folder):
        os.makedirs(submission_folder)

    # Description file
    # description_file = os.path.join(submission_folder, 'description.txt')
    # lines = []
    # lines.append('method name: KPConv\n')
    # lines.append('method description: Fully convolutional network operating directly on points\n')
    # lines.append('project url: https://github.com/HuguesTHOMAS/KPConv\n')
    # lines.append('publication url: https://arxiv.org/abs/1904.08889\n')
    # lines.append('bibtex: @article{thomas2019KPConv, \
    # Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz \
    # and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.}, \
    # Title = {KPConv: Flexible and Deformable Convolution for Point Clouds}, \
    # Journal = {Proceedings of the IEEE International Conference on Computer Vision}, \
    # Year = {2019} }\n')
    # lines.append('organization or affiliation: Mines Paristech\n')
    # with open(description_file, 'w') as f:
    #     f.writelines(lines)

    for seq_name, seq_frames in zip(dataset.test_sequences, dataset.test_frames):

        # Create sequence folder
        seq_folder = os.path.join(submission_folder, seq_name, 'predictions')
        if not os.path.exists(seq_folder):
            os.makedirs(seq_folder)

        # Advanced display
        N = len(seq_frames)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nSequence ' + seq_name)

        for f_i, frame_name in enumerate(seq_frames):

            # Name of prediction file
            filename ='{:s}_{:07d}.npy'.format(seq_name, f_i)
            filepath = os.path.join(test_path, 'probs', filename)

            # Load files
            if os.path.exists(filepath):
                frame_probs = np.load(filepath)
            else:
                raise ValueError(filepath + ' not found. Ending script')

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(dataset.label_values):
                if label_value in dataset.ignored_labels:
                    frame_probs = np.insert(frame_probs, l_ind, 0, axis=1)

            # Predicted labels
            frame_preds = dataset.label_values[np.argmax(frame_probs, axis=1)]

            # Convert to official labels
            frame_preds = dataset.learning_map_inv[frame_preds].astype(np.uint32)

            # Save to binary file
            submission_file = os.path.join(seq_folder, '{:s}.label'.format(frame_name))
            frame_preds.tofile(submission_file)

            print('', end='\r')
            print(fmt_str.format('#' * ((f_i * progress_n) // N), 100 * f_i / N),
                  end='',
                  flush=True)

        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)