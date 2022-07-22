#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
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
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd
from sklearn.metrics import confusion_matrix

# My libs
from utils.config import Config, MultiConfig
from utils.metrics import IoU_from_confusions, smooth_metrics
from utils.ply import read_ply

# Datasets
from datasets.ModelNet40 import ModelNet40Dataset
from datasets.ShapeNetPart import ShapeNetPartDataset
from datasets.S3DIS import S3DISDataset
from datasets.Scannet import ScannetDataset
from datasets.Semantic3D import Semantic3DDataset
from datasets.NPM3D import NPM3DDataset
from datasets.SemanticKitti import SemanticKittiDataset

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def running_mean(signal, n, axis=0):

    signal = np.array(signal)
    if signal.ndim == 1:
        signal_sum = np.convolve(signal, np.ones((2*n+1,)), mode='same')
        signal_num = np.convolve(signal*0+1, np.ones((2*n+1,)), mode='same')
        return signal_sum/signal_num

    elif signal.ndim == 2:
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_multi_metrics(all_IoUs, smooth_n):

    # Get mean IoU for consecutive epochs to directly get a mean
    all_mIoUs = [np.hstack([np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs]) for epoch_IoUs in all_IoUs]
    smoothed_mIoUs = []
    for epoch in range(len(all_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_mIoUs))
        smoothed_mIoUs += [np.mean(np.hstack(all_mIoUs[i0:i1]))]

    # Get mean for each class
    all_objs_mIoUs = [[np.mean(obj_IoUs, axis=1) for obj_IoUs in epoch_IoUs] for epoch_IoUs in all_IoUs]
    smoothed_obj_mIoUs = []
    for epoch in range(len(all_objs_mIoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_objs_mIoUs))

        epoch_obj_mIoUs = []
        for obj in range(len(all_objs_mIoUs[0])):
            epoch_obj_mIoUs += [np.mean(np.hstack([objs_mIoUs[obj] for objs_mIoUs in all_objs_mIoUs[i0:i1]]))]

        smoothed_obj_mIoUs += [epoch_obj_mIoUs]

    return np.array(smoothed_mIoUs), np.array(smoothed_obj_mIoUs)


def IoU_class_metrics(all_IoUs, smooth_n):

    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_report_confusion(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = lines[6:6+n_class]

    conf = []
    for i, line in enumerate(lines):
        conf.append(np.array([float(value) for value in line.split()]))

    return np.vstack(conf).astype(np.int32)


def plot_test_stats(dataset, log):

    # Parameters
    # **********

    # Smoothing
    smooth_n = 10

    # Load parameters
    #config = Config()
    #config.load(log)

    # Report folder
    report_path = join('test', log, 'reports')

    # Name of each class
    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    # Read reports
    # ************

    # All report pathes
    report_names = [join(report_path, f) for f in listdir(report_path) if 'report' in f]
    report_names = np.sort(report_names)

    # Test epochs and confs
    test_epochs = [int(f[:-4].split('_')[-1]) for f in report_names]
    test_confs = np.stack([load_report_confusion(f, dataset.num_classes) for f in report_names], axis=0)

    # Plot tested ratio
    # *****************

    # Get percentage of predicted points
    tested_ratio = (1 - test_confs[:, 1:, 0] / np.sum(test_confs[:, 1:, :], axis=2)) * 100

    # Figure
    plt.figure('tested_ratio')
    for i, name in enumerate(class_list):
        plt.plot(test_epochs, tested_ratio[:, i], linewidth=1, label=name)
    plt.xlabel('epochs')
    plt.ylabel('ratio')

    # Display legends and title
    plt.legend(loc=4)

    # Plot IoU
    # ********

    test_confs = test_confs[:, 1:, 1:]
    IoUs = IoU_from_confusions(test_confs)

    # Figure
    plt.figure('IoUs')
    for i, name in enumerate(class_list):
        plt.plot(test_epochs, IoUs[:, i], linewidth=1, label=name)
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Display legends and title
    plt.legend(loc=4)

    # Figure
    plt.figure('mIoU')
    plt.plot(test_epochs, np.mean(IoUs, axis=1), linewidth=1)
    plt.xlabel('epochs')
    plt.ylabel('mIoU')

    # Display legends and title
    plt.legend(loc=4)

    # Show all
    plt.show()

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # My logs: choose the logs to show
    log = 'Log_2019-11-07_19-28-12'

    # Load dataset
    dataset = SemanticKittiDataset(n_frames=5)
    plot_test_stats(dataset, log)




