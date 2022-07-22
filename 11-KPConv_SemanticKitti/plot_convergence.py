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


def load_confusions(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


def load_training_results(path):

    filename = join(path, 'training.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    steps = []
    L_out = []
    L_reg = []
    L_p = []
    acc = []
    t = []
    memory = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            steps += [int(line_info[0])]
            L_out += [float(line_info[1])]
            L_reg += [float(line_info[2])]
            L_p += [float(line_info[3])]
            acc += [float(line_info[4])]
            t += [float(line_info[5])]
            memory += [float(line_info[6])]
        else:
            break

    return steps, L_out, L_reg, L_p, acc, t, memory


def load_single_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds(path, dataset, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    Confs[c_i] += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_snap_clouds(path, dataset, file_i, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    if len(cloud_folders) > 0:
        dataset_folders = [f for f in listdir(cloud_folders[0]) if dataset.name in f]
        cloud_folders = [join(f, dataset_folders[file_i]) for f in cloud_folders]

    Confs = np.zeros((len(cloud_epochs), dataset.num_classes, dataset.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf_{:s}.txt'.format(dataset.name))
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    if np.any([cloud_path.endswith(f) for cloud_path in dataset.train_files]):
                        data = read_ply(join(cloud_folder, f))
                        labels = data['class']
                        preds = data['preds']
                        Confs[c_i] += confusion_matrix(labels, preds, dataset.label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
        if label_value in dataset.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_multi_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        obj_IoUs = [[float(IoU) for IoU in s.split()] for s in line.split('/')]
        obj_IoUs = [np.reshape(IoUs, [-1, n_parts[obj]]) for obj, IoUs in enumerate(obj_IoUs)]
        all_IoUs += [obj_IoUs]
    return all_IoUs


def compare_trainings(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_epochs = 1

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    all_epochs = []
    all_loss = []
    all_lr = []
    all_times = []
    all_RAMs = []

    for path in list_of_paths:

        if ('val_IoUs.txt' in [f for f in listdir(path)]) or ('val_confs.txt' in [f for f in listdir(path)]):
            config = Config()
            config.load(path)
        elif np.any(['val_IoUs' in f for f in listdir(path)]):
            config = MultiConfig()
            config.load(path)
        else:
            continue

        # Compute number of steps per epoch
        if config.epoch_steps is None:
            if config.dataset == 'ModelNet40':
                steps_per_epoch = np.ceil(9843 / int(config.batch_num))
            else:
                raise ValueError('Unsupported dataset')
        else:
            steps_per_epoch = config.epoch_steps

        smooth_n = int(steps_per_epoch * smooth_epochs)

        # Load results
        steps, L_out, L_reg, L_p, acc, t, memory = load_training_results(path)
        all_epochs += [np.array(steps) / steps_per_epoch]
        all_loss += [running_mean(L_out, smooth_n)]
        all_times += [t]
        all_RAMs += [memory]

        # Learning rate
        lr_decay_v = np.array([lr_d for ep, lr_d in config.lr_decays.items()])
        lr_decay_e = np.array([ep for ep, lr_d in config.lr_decays.items()])
        max_e = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
        lr_decays = np.ones(int(np.ceil(max_e)), dtype=np.float32)
        lr_decays[0] = float(config.learning_rate)
        lr_decays[lr_decay_e] = lr_decay_v
        lr = np.cumprod(lr_decays)
        all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

    # Plots learning rate
    # *******************

    if False:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    if True:
        # Figure
        fig = plt.figure('RAM')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_RAMs[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('RAM')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))


    # Plots loss
    # **********

    # Figure
    fig = plt.figure('loss')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')

    # Display legends and title
    plt.legend(loc=1)
    plt.title('Losses compare')

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plot Times
    # **********

    # Figure
    fig = plt.figure('time')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_epochs[i], np.array(all_times[i]) / 3600, linewidth=1, label=label)

    # Set names for axes
    plt.xlabel('epochs')
    plt.ylabel('time')
    # plt.yscale('log')

    # Display legends and title
    plt.legend(loc=0)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_multisegment(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 5

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_instances_mIoUs = []
    all_objs_mIoUs = []
    all_objs_IoUs = []
    all_parts = []

    obj_list = ['Air', 'Bag', 'Cap', 'Car', 'Cha', 'Ear', 'Gui', 'Kni',
                'Lam', 'Lap', 'Mot', 'Mug', 'Pis', 'Roc', 'Ska', 'Tab']
    print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
    print('-----|------|--------------------------------------------------------------------------------')
    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(path)

        # Get the number of classes
        n_parts = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        part = config.dataset.split('_')[-1]

        # Get validation confusions
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_multi_IoU(file, n_parts)

        file = join(path, 'vote_IoUs.txt')
        vote_IoUs = load_multi_IoU(file, n_parts)

        #print(len(val_IoUs[0]))
        #print(val_IoUs[0][0].shape)

        # Get mean IoU
        #instances_mIoUs, objs_mIoUs = IoU_multi_metrics(val_IoUs, smooth_n)

        # Get mean IoU
        instances_mIoUs, objs_mIoUs = IoU_multi_metrics(vote_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_instances_mIoUs += [instances_mIoUs]
        all_objs_IoUs += [objs_mIoUs]
        all_objs_mIoUs += [np.mean(objs_mIoUs, axis=1)]

        if part == 'multi':
            s = '{:4.1f} | {:4.1f} | '.format(100 * np.mean(objs_mIoUs[-1]), 100 * instances_mIoUs[-1])
            for obj_mIoU in objs_mIoUs[-1]:
                s += '{:4.1f} '.format(100 * obj_mIoU)
            print(s)
        else:
            s = ' --  |  --  | '
            for obj_name in obj_list:
                if part.startswith(obj_name):
                    s += '{:4.1f} '.format(100 * instances_mIoUs[-1])
                else:
                    s += ' --  '.format(100 * instances_mIoUs[-1])
            print(s)
        all_parts += [part]

    # Plots
    # *****

    if 'multi' in all_parts:

        # Figure
        fig = plt.figure('Instances mIoU')
        for i, label in enumerate(list_of_labels):
            if all_parts[i] == 'multi':
                plt.plot(all_pred_epochs[i], all_instances_mIoUs[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel('IoU')

        # Set limits for y axis
        #plt.ylim(0.55, 0.95)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

        # Figure
        fig = plt.figure('mean of categories mIoU')
        for i, label in enumerate(list_of_labels):
            if all_parts[i] == 'multi':
                plt.plot(all_pred_epochs[i], all_objs_mIoUs[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel('IoU')

        # Set limits for y axis
        #plt.ylim(0.8, 1)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    for obj_i, obj_name in enumerate(obj_list):
        if np.any([part.startswith(obj_name) for part in all_parts]):
            # Figure
            fig = plt.figure(obj_name + ' mIoU')
            for i, label in enumerate(list_of_labels):
                if all_parts[i] == 'multi':
                    plt.plot(all_pred_epochs[i], all_objs_IoUs[i][:, obj_i], linewidth=1, label=label)
                elif all_parts[i].startswith(obj_name):
                    plt.plot(all_pred_epochs[i], all_objs_mIoUs[i], linewidth=1, label=label)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()


def compare_convergences_segment(dataset, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_mIoUs = []
    all_class_IoUs = []
    all_snap_epochs = []
    all_snap_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10*'-' + '|' + 10*config.num_classes*'-')
    for path in list_of_paths:

        # Get validation IoUs
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, config.num_classes)

        # Get mean IoU
        class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_mIoUs += [mIoUs]
        all_class_IoUs += [class_IoUs]

        s = '{:^10.1f}|'.format(100*mIoUs[-1])
        for IoU in class_IoUs[-1]:
            s += '{:^10.1f}'.format(100*IoU)
        print(s)

        # Get optional full validation on clouds
        snap_epochs, snap_IoUs = load_snap_clouds(path, dataset)
        all_snap_epochs += [snap_epochs]
        all_snap_IoUs += [snap_IoUs]

    print(10*'-' + '|' + 10*config.num_classes*'-')
    for snap_IoUs in all_snap_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^10.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^10.1f}'.format(100*IoU)
        else:
            s = '{:^10s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^10s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()


def compare_convergences_classif(dataset, list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 2

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_OA = []
    all_train_OA = []
    all_vote_OA = []
    all_vote_confs = []


    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(list_of_paths[0])

        # Get the number of classes
        n_class = config.num_classes

        # Get validation confusions
        file = join(path, 'val_confs.txt')
        val_C1 = load_confusions(file, n_class)
        val_PRE, val_REC, val_F1, val_IoU, val_ACC = smooth_metrics(val_C1, smooth_n=smooth_n)

        # Get vote confusions
        file = join(path, 'vote_confs.txt')
        if exists(file):
            vote_C2 = load_confusions(file, n_class)
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = smooth_metrics(vote_C2, smooth_n=2)
        else:
            vote_C2 = val_C1
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = (val_PRE, val_REC, val_F1, val_IoU, val_ACC)

        # Get training confusions balanced
        file = join(path, 'training_confs.txt')
        train_C = load_confusions(file, n_class)
        train_PRE, train_REC, train_F1, train_IoU, train_ACC = smooth_metrics(train_C, smooth_n=smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_ACC))])]
        all_val_OA += [val_ACC]
        all_vote_OA += [vote_ACC]
        all_train_OA += [train_ACC]
        all_vote_confs += [vote_C2]
        #all_mean_IoU_scores += [running_mean(np.mean(val_IoU[:, 1:], axis=1), smooth_n)]

    print()


    # Best scores
    # ***********

    for i, label in enumerate(list_of_labels):

        print('\n' + label + '\n' + '*' * len(label) + '\n')
        print(list_of_paths[i])

        best_epoch = np.argmax(all_vote_OA[i])
        print('Best Accuracy : {:.1f} % (epoch {:d})'.format(100 * all_vote_OA[i][best_epoch], best_epoch))

        confs = all_vote_confs[i]

        """
        s = ''
        for cc in confs[best_epoch]:
            for c in cc:
                s += '{:.0f} '.format(c)
            s += '\n'
        print(s)
        """

        TP_plus_FN = np.sum(confs, axis=-1, keepdims=True)
        class_avg_confs = confs.astype(np.float32) / TP_plus_FN.astype(np.float32)
        diags = np.diagonal(class_avg_confs, axis1=-2, axis2=-1)
        class_avg_ACC = np.sum(diags, axis=-1) / np.sum(class_avg_confs, axis=(-1, -2))

        print('Corresponding mAcc : {:.1f} %'.format(100 * class_avg_ACC[best_epoch]))

    # Plots
    # *****

    # Figure
    fig = plt.figure('Validation')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_pred_epochs[i], all_val_OA[i], linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.ylabel('Validation Accuracy')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Figure
    fig = plt.figure('Vote Validation')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_pred_epochs[i], all_vote_OA[i], linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.ylabel('Validation Accuracy')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Figure
    fig = plt.figure('Training')
    for i, label in enumerate(list_of_labels):
        plt.plot(all_pred_epochs[i], all_train_OA[i], linewidth=1, label=label)
    plt.xlabel('epochs')
    plt.ylabel('Overall Accuracy')

    # Set limits for y axis
    #plt.ylim(0.8, 1)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))


    #for i, label in enumerate(list_of_labels):
    #    print(label, np.max(all_train_OA[i]), np.max(all_val_OA[i]))


    # Show all
    plt.show()


def compare_convergences_multicloud(list_of_paths, multi, multi_datasets, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]


    # Loop on all datasets:
    for plot_dataset in multi_datasets:
        print('\n')
        print(plot_dataset)
        print('*'*len(plot_dataset))
        print()

        # Load dataset parameters
        if plot_dataset.startswith('S3DIS'):
            dataset = S3DISDataset()
        elif plot_dataset.startswith('Scann'):
            dataset = ScannetDataset()
        elif plot_dataset.startswith('Seman'):
            dataset = Semantic3DDataset()
        elif plot_dataset.startswith('NPM3D'):
            dataset = NPM3DDataset()
        else:
            raise ValueError('Unsupported dataset : ' + plot_dataset)

        # Read Logs
        # *********

        all_pred_epochs = []
        all_mIoUs = []
        all_class_IoUs = []
        all_snap_epochs = []
        all_snap_IoUs = []
        all_names = []

        class_list = [dataset.label_to_names[label] for label in dataset.label_values
                      if label not in dataset.ignored_labels]

        s = '{:^10}|'.format('mean')
        for c in class_list:
            s += '{:^10}'.format(c)
        print(s)
        print(10*'-' + '|' + 10*dataset.num_classes*'-')
        for log_i, (path, is_multi) in enumerate(zip(list_of_paths, multi)):

            n_c = None
            if is_multi:
                config = MultiConfig()
                config.load(path)
                if plot_dataset in config.datasets:
                    val_IoU_files = []
                    for d_i in np.where(np.array(config.datasets) == plot_dataset)[0]:
                        n_c = config.num_classes[d_i]
                        val_IoU_files.append(join(path, 'val_IoUs_{:d}_{:s}.txt'.format(d_i, plot_dataset)))
                else:
                    continue
            else:
                config = Config()
                config.load(path)
                if plot_dataset == config.dataset:
                    n_c = config.num_classes
                    val_IoU_files = [join(path, 'val_IoUs.txt')]
                else:
                    continue

            for file_i, file in enumerate(val_IoU_files):

                # Load validation IoUs
                val_IoUs = load_single_IoU(file, n_c)

                # Get mean IoU
                class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

                # Aggregate results
                all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
                all_mIoUs += [mIoUs]
                all_class_IoUs += [class_IoUs]
                all_names += [list_of_names[log_i]+'_{:d}'.format(file_i+1)]

                s = '{:^10.1f}|'.format(100*mIoUs[-1])
                for IoU in class_IoUs[-1]:
                    s += '{:^10.1f}'.format(100*IoU)
                print(s)

                # Get optional full validation on clouds
                if is_multi:
                    snap_epochs, snap_IoUs = load_multi_snap_clouds(path, dataset, file_i)
                else:
                    snap_epochs, snap_IoUs = load_snap_clouds(path, dataset)
                all_snap_epochs += [snap_epochs]
                all_snap_IoUs += [snap_IoUs]

        print(10*'-' + '|' + 10*dataset.num_classes*'-')
        for snap_IoUs in all_snap_IoUs:
            if len(snap_IoUs) > 0:
                s = '{:^10.1f}|'.format(100*np.mean(snap_IoUs[-1]))
                for IoU in snap_IoUs[-1]:
                    s += '{:^10.1f}'.format(100*IoU)
            else:
                s = '{:^10s}'.format('-')
                for _ in range(dataset.num_classes):
                    s += '{:^10s}'.format('-')
            print(s)

        # Plots
        # *****

        # Figure
        fig = plt.figure('mIoUs')
        for i, name in enumerate(all_names):
            p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
            plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())

        plt.title(plot_dataset)
        plt.xlabel('epochs')
        plt.ylabel('IoU')

        # Set limits for y axis
        #plt.ylim(0.55, 0.95)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

        displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
        displayed_classes = []
        for c_i, c_name in enumerate(class_list):
            if c_i in displayed_classes:

                # Figure
                fig = plt.figure(c_name + ' IoU')
                for i, name in enumerate(list_of_names):
                    plt.plot(all_pred_epochs[i], all_class_IoUs[i][:, c_i], linewidth=1, label=name)
                plt.xlabel('epochs')
                plt.ylabel('IoU')

                # Set limits for y axis
                #plt.ylim(0.8, 1)

                # Display legends and title
                plt.legend(loc=4)

                # Customize the graph
                ax = fig.gca()
                ax.grid(linestyle='-.', which='both')
                #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



        # Show all
        plt.show()


def compare_convergences_SLAM(dataset, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_mIoUs = []
    all_val_class_IoUs = []
    all_subpart_mIoUs = []
    all_subpart_class_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10*'-' + '|' + 10*config.num_classes*'-')
    for path in list_of_paths:

        # Get validation IoUs
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, config.num_classes)

        # Get Subpart IoUs
        file = join(path, 'subpart_IoUs.txt')
        subpart_IoUs = load_single_IoU(file, config.num_classes)

        # Get mean IoU
        val_class_IoUs, val_mIoUs = IoU_class_metrics(val_IoUs, smooth_n)
        subpart_class_IoUs, subpart_mIoUs = IoU_class_metrics(subpart_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_val_mIoUs += [val_mIoUs]
        all_val_class_IoUs += [val_class_IoUs]
        all_subpart_mIoUs += [subpart_mIoUs]
        all_subpart_class_IoUs += [subpart_class_IoUs]

        s = '{:^10.1f}|'.format(100*subpart_mIoUs[-1])
        for IoU in subpart_class_IoUs[-1]:
            s += '{:^10.1f}'.format(100*IoU)
        print(s)


    print(10*'-' + '|' + 10*config.num_classes*'-')
    for snap_IoUs in all_val_class_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^10.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^10.1f}'.format(100*IoU)
        else:
            s = '{:^10s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^10s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_subpart_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_pred_epochs[i], all_val_mIoUs[i], linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_val_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()




# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


def scannet_small_in_radius():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2018-12-21_17-01-53'
    end = 'Log_2019-01-15_13-22-17'
    logs = np.sort([join('../4-KP-R-CNN/results', l) for l in listdir('../4-KP-R-CNN/results') if start <= l <= end])

    print(listdir('../4-KP-R-CNN/results'))

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.100_R1.5_1F',
                  'k15_d2.5_r0.075_R1.2_5F',
                  'k15_d2.5_r0.100_R2.0_1F',
                  'k15_d2.5_r0.075_R1.5_1F',
                  'k15_d2.5_r0.075_R1.5_2F',
                  'k15_d2.5_r0.075_R1.5_1F_closest',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def scannet_full_rooms():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-01-16_15-22-17'
    end = 'Log_2019-01-20_13-22-17'
    logs = np.sort([join('../4-KP-R-CNN/results', l) for l in listdir('../4-KP-R-CNN/results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.125_Rinf_1F_closest',
                  'k15_d2.5_r0.125_Rinf_4F_closest',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet_deeper_nets():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-01-20_13-22-17'
    end = 'Log_2019-01-30_13-22-17'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.075',
                  'k15_d2.0_r0.06_deeper',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def scannet_deeper_nets():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-01-25_13-22-17'
    end = 'Log_2019-01-30_13-22-17'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.hstack(('../4-KP-R-CNN/results/Log_2019-01-14_14-07-42', logs))

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.075_R1.5_1F_closest',
                  'k15_d2.0_r0.06_R1.5_1F_closest',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def npm3d():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-02-01_13-22-17'
    end = 'Log_2019-02-10_13-22-17'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.hstack((logs, 'results/Log_2019-03-20_23-21-31'))

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.10_R2.5_1F_closest',
                  'k15_d2.5_r0.075_R2.0_1F_closest',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def semantic3D():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-02-15_20-22-17'
    end = 'Log_2019-02-20_13-22-17'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.10_R2.0_4F_closest',
                  'k15_d2.5_r0.15_R3.0_4F_closest',
                  'k15_d2.5_r0.20_R4.0_4F_closest',
                  'k15_d2.5_r0.25_R5.0_4F_closest',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def semantic3D_alternate_validation_set():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-02-20_13-22-17'
    end = 'Log_2019-02-21_13-22-17'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.10_R2.0_4F_closest',
                  'k15_d2.5_r0.15_R3.0_4F_closest',
                  'k15_d2.5_r0.20_R4.0_4F_closest',
                  'k15_d2.5_r0.25_R5.0_4F_closest',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def semantic3D_color_augmentation():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-02-21_13-22-17'
    end = 'Log_2019-02-22_23-22-17'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.hstack((logs, 'results/Log_2019-02-16_12-19-20'))


    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.15_R3.0_4F_closest_color0.5',
                  'k15_d2.5_r0.15_R3.0_4F_closest_color0.7',
                  'k15_d2.5_r0.15_R3.0_4F_closest_color0.9',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_small():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-02_11-21-30'
    end = 'Log_2019-03-02_19-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])


    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.1_1F_small',
                  'k15_d2.5_r0.1_4F_small',
                  'k15_d2.5_r0.1_5F_small',
                  'k15_d2.5_r0.1_4F_small_nosym',
                  'k15_d2.5_r0.1_1F_small_nosym',
                  'k15_d2.5_r0.075_4F_nosym',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_full():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-03_00-21-30'
    end = 'Log_2019-03-05_16-59-30'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])


    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_r0.05_1F_sym',
                  'k15_d2.5_r0.05_1F_nosym',
                  'k15_d2.5_r0.05_4F_nosym',
                  'k15_d2.5_r0.05_5F_nosym',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ModelNet40_deformable():


    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-05_16-20-30'
    end = 'Log_2019-03-06_12-24-20'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['k15_d2.5_l0.04_r0.1_old_closest',
                  'k15_d2.5_l0.04_Kext1.0_constant_closest',
                  'k15_d2.5_l0.04_Kext1.0_constant_sum',
                  'NORMAL_k15_d2.5_l0.04_Kext1.0_linear_closest',
                  'NORMAL_k15_d2.5_l0.04_Kext1.0_linear_sum',
                  'k15_d2.5_l0.04_Kext1.0_gaussian_closest',
                  'k15_d2.5_l0.04_Kext1.0_gaussian_sum',
                  'DEFORM_k15_d4.0_l0.04_Kext1.0_linear_closest',
                  'DEFORM_k15_d4.0_l0.04_Kext1.0_linear_sum',
                  'DEFORM_k15_d4.0_l0.04_Kext1.0_linear_sum_4L',
                  'NORMAL_k15_d4.0_l0.04_Kext1.0_linear_sum_4L',
                  'MDEFOR_k15_d4.0_l0.04_Kext1.0_linear_sum_4L',
                  'DEFORM_k15_d4.0_l0.04_Kext1.0_linear_sum_4L_fitting',
                  'DEFORM_k15_d4.0_l0.04_Kext1.0_linear_sum_4L_permissive',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    # Remove useless
    useless = [0, 1, 2, 5, 6]
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)

    return logs, logs_names


def ModelNet40_deformable_full():
    """
    Experiment showing that deformable conv perform slightly better than normal conv in the same configuration (linear
    sum). Although the old closest achieved an equal score, we do not have to show it.
    :return:
    """


    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-06_15-10-45'
    end = 'Log_2019-03-06_18-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.hstack(('results/Log_2019-03-03_11-31-53', 'results/Log_2019-03-07_11-11-34', logs))
    logs = np.insert(logs, 3, 'results/Log_2019-09-24_19-11-27')

    # Give names to the logs (for legends)
    logs_names = ['NORMAL_k15_d2.5_l0.02_old_closest',
                  'NORMAL_k15_d4.0_l0.02_Kext1.0_linear_sum',
                  'DEFORM3_k15_d4.0_l0.02_Kext1.0_linear_sum',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def Semantic3D_deformable():
    """
    Conclusion de cette experience, on voit que le fitting loss a l'air d'ameliorer les choses mais toujours en dessous
    de KPConv normal dans la meme configuration. On va essayer sur un autre dataset et on va augmenter la force du
    fitting loss car il y a toujours des KP lost dans quand offset_decay = 0.01
    :return:
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-06_18-35-18'
    end = 'Log_2019-03-13_11-56-40'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.delete(logs, [1])
    logs = np.hstack(('results/Log_2019-02-21_13-16-20', logs))

    # Give names to the logs (for legends)
    logs_names = ['NORMAL_k15_d2.5_l0.06_R3.0_old_closest',
                  'NORMAL_k15_d4.0_l0.06_Kext1.0_R3.0_linear_sum',
                  'DEFORM3_k15_d4.0_l0.06_Kext1.0_R3.0_linear_sum',
                  'MDEFORM3_k15_d4.0_l0.06_Kext1.0_R3.0_linear_sum',
                  'DEFORM5_same_(spatial_b_gen)(grad*1.0)',
                  'DEFORM5_same_(spatial_b_gen)(grad*1.0)(fitting_loss)',
                  'DEFORM5_same_(spatial_b_gen)',
                  'DEFORM5_same_(spatial_b_gen)(fitting_loss)',
                  'DEFORMALL_same_(spatial_b_gen)(fitting_loss)',
                  'NORMAL_same_(spatial_b_gen)',
                  'DEFORM5_same_(spatial_b_gen)(fitting_loss+)',
                  'DEFORM5_same_(spatial_b_gen)(fitting_smart+)',
                  'DEFORM5_inception_deformable(permissive)',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])

    # Remove useless
    useless = [0, 1, 2, 3, 4, 5, 7, 8, 10]
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def Scannet_deformable():
    """

    :return:
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-13_11-56-41'
    end = 'Log_2019-03-15_15-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['NORMAL',
                  'DEFORM_5',
                  'INCEPTION_5',
                  'M_INCEPTION_5',
                  'restricted_NORMAL',
                  'restricted_DEFORM_3',
                  'restricted_DEFORM_5',
                  'restricted_MDEFORM_5',
                  'restricted_MINCEPTION_5',
                  'test',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = [5, 8, 9, 10]
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def Scannet_restricted():
    """

    :return:
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-17_09-29-21'
    end = 'Log_2019-03-17_23-46-59'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.hstack((logs, 'results/Log_2019-03-13_20-10-43', 'results/Log_2019-03-22_12-59-15'))

    # Give names to the logs (for legends)
    logs_names = ['DEFORM_5_KP4',
                  'DEFORM_5_KP5',
                  'DEFORM_5_KP7',
                  'DEFORM_5_KP9',
                  'DEFORM_5_KP11',
                  'DEFORM_5_KP15',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def Scannet_restricted_bis():
    """

    :return:
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-18_14-43-59'
    end = 'Log_2019-03-19_13-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.hstack((logs, 'results/Log_2019-03-13_19-11-27', 'results/Log_2019-03-22_14-08-52'))

    # Give names to the logs (for legends)
    logs_names = ['NORMAL_KP4',
                  'NORMAL_KP5',
                  'NORMAL_KP7',
                  'NORMAL_KP9',
                  'NORMAL_KP15',
                  'test',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def S3DIS_deform():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-19_19-14-24'
    end = 'Log_2019-03-19_20-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['NORMAL_KP15',
                  'DEFORM_KP15',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def ShapeNetPart_deform():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-20_15-14-24'
    end = 'Log_2019-03-21_20-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['NORMAL_KP15_batch32',
                  'DEFORM_KP15_batch32_scale10%',
                  'DEFORM_KP15_batch16_scale20%',
                  'DEFORM_KP15_batch16_scale20%_KPext1.2',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def S3DIS_k_fold():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-23_19-14-24'
    end = 'Log_2019-03-24_20-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 4, 'results/Log_2019-03-19_19-14-57')

    # Give names to the logs (for legends)
    logs_names = ['DEFORM_Area1',
                  'DEFORM_Area2',
                  'DEFORM_Area3',
                  'DEFORM_Area4',
                  'DEFORM_Area5',
                  'DEFORM_Area6',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def S3DIS_k_fold_bis():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-24_19-14-24'
    end = 'Log_2019-03-27_13-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 4, 'results/Log_2019-03-19_19-14-24')

    # Give names to the logs (for legends)
    logs_names = ['NORMAL_Area1',
                  'NORMAL_Area2',
                  'NORMAL_Area3',
                  'NORMAL_Area4',
                  'NORMAL_Area5',
                  'NORMAL_Area6',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def NPM3D_deformable():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-27_20-32-03'
    end = 'Log_2019-03-28_20-43-50'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['DEFORM_spatial_gen_K15',
                  'DEFORM_spatial_gen_K20',
                  'DEFORM_spatial_gen_K30',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def Scannet_more():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-03-28_20-43-51'
    end = 'Log_2019-04-30_10-16-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['DEFORM_dl.020*50',
                  'DEFORM_dl.025*50',
                  'DEFORM_dl.030*50',
                  'DEFORM_dl.040*50_epoch1000',
                  'DEFORM_dl.050*50_epoch1000',
                  'DEFORM_dl.040*40_epoch500',
                  'DEFORM_dl.050*40_epoch500',
                  'DEFORM_dl.040*60_epoch500',
                  'DEFORM_dl.030*60_epoch500',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def S3DIS_new():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-06-06_12-49-00'
    end = 'Log_2019-06-19_20-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 2, 'results/Log_2019-03-27_20-32-03')
    logs = np.insert(logs, 3, 'results/Log_2019-03-13_10-21-04')
    logs = np.insert(logs, 4, 'results/Log_2019-03-13_20-10-43')

    # Give names to the logs (for legends)
    logs_names = ['S3DIS_NORMAL_KP15',
                  'S3DIS_DEFORM_KP15',
                  'NPM3D_DEFORM_KP15',
                  'Semantic3D_DEFORM_KP15',
                  'Scannet_DEFORM_KP15',
                  'multi_DEFORM_KP15',
                  'multi_DEFORM_KP15_bis',
                  'multi_DEFORM_KP15_tris',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def ModelNet40_pretrained():


    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-06-26_16-38-27'
    end = 'Log_2019-06-29_06-38-27'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 0, 'results/Log_2019-03-06_15-12-44')

    # Give names to the logs (for legends)
    logs_names = ['DEFORM3_untrained_L1e-3_d80',
                  'DEFORM5_L1e-4_d80_all_layer',
                  'DEFORM5_L1e-3_d20_head+4',
                  'RIGID_untrained_L1e-3_d80',
                  'DEFORM5_untrained_L1e-3_d80',
                  'RIGID_untrained_flips',
                  'RIGID_untrained_rots',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def ShapeNetPart_pretrained():


    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-06-29_06-38-27'
    end = 'Log_2019-07-01_06-38-27'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 0, 'results/Log_2019-03-21_12-17-11')
    logs = np.insert(logs, 1, 'results/Log_2019-03-21_12-17-48')

    # Give names to the logs (for legends)
    logs_names = ['DEFORM5_old',
                  'DEFORM5_old_KPext1.2',
                  'DEFORM5_new',
                  'DEFORM5_new_pretrained',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def multi_comparison():

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-07-11_10-49-00'
    end = 'Log_2019-07-19_20-26-43'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 0, 'results/Log_2019-06-06_12-49-00')
    logs = np.insert(logs, 1, 'results/Log_2019-03-27_20-32-03')
    logs = np.insert(logs, 2, 'results/Log_2019-03-13_10-21-04')
    logs = np.insert(logs, 3, 'results/Log_2019-03-13_20-10-43')

    # Give names to the logs (for legends)
    logs_names = ['S3DIS_DEFORM_dl0=0.04',
                  'NPM3D_DEFORM_KP15',
                  'Semantic3D_DEFORM_KP15 NOT SAME SPLIT',
                  'Scannet_DEFORM_KP15',
                  'multi(4)+S3DIS_DEFORM_dl0=0.03',
                  'multi(1)+S3DIS_DEFORM_dl0=0.03',
                  'multi(1)+Sema3D_DEFORM_dl0=0.06',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def SemanticKitti_first():
    """
    First test with SemanticKitti. With a 3D KPConv on multiple frames.
    The three first tries are done with in_feature = 1 and the other in_feature = 4
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-09-05_17-44-39'
    end = 'Log_2019-09-24_16-44-38'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    #logs = np.hstack((logs, 'results/Log_2019-03-13_19-11-27', 'results/Log_2019-03-22_14-08-52'))

    # Give names to the logs (for legends)
    logs_names = ['RIGID_d100',
                  'RIGID_d150',
                  'DEFORM_d100',
                  'DEFORM_d100_XYZ',
                  'SAME_dl012_max80K_(merge_consecutive)',
                  'RIGID_dl010_max120K_(merge_consecutive)',
                  'SAME_(merge_distance)',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def ModelNet40_optimization():


    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-09-24_16-38-27'
    end = 'Log_2019-09-29_06-38-27'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])
    logs = np.insert(logs, 0, 'results/Log_2019-03-06_15-12-44')

    # Give names to the logs (for legends)
    logs_names = ['DEFORM3_L1e-3_d80_full_epoch',
                  'RIGID_L1e-3_d100_epoch300',
                  'RIGID_L1e-2_d50_epoch200',
                  'SAME_random_balanced',
                  'SAME_potential_balanced',
                  'SAME_d100_potential_balanced',
                  'SAME_d100_potential_all',
                  'DEFORM5_L1e-3_d70_epoch300',
                  'DEFORM5_L1e-2_d70_epoch300',
                  'DEFORM5_L1e-2_d100_epoch300',
                  'DEFORM5_L1e-2_d150_epoch300',
                  'DEFORM5_L1e-2_d150_epoch300(augmsym)',
                  'test']

    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def SemanticKitti_soloframe():
    """
    Restart experiments on semantic Kitti with soloframe to see if something is wrong with some classes and the
    validation.
    A partir du 3eme, correction du class balanced qui decallait toutes les calsses de 1. Aussi, ajout du pick du
    center point sur la bonne classe quand radius < 50.0. Ajout de la sauvegarde des validation preds. Correction de
    la fonction de validation pour qu'elle fonctionne quand radius < 50.0. Tout ca est a verifier pour n_frames > 1
    A partir de la quatrieme, batch gen de la validation utilise des potentiels dans chaque frames
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-10-03_18-38-23'
    end = 'Log_2019-10-24_16-44-38'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['RIGID_d100_e1000_1',
                  'RIGID_d100_e1000_1XYZ',
                  'RIGID_dl0.06_R6.0',
                  'DEFORM_dl0.06_R6.0',
                  'RIGID_dl0.06_R6.0',
                  'DEFORM_dl0.04_R4.0',
                  'DEFORM_dl0.10_R10.0',
                  'RIGID5_dl0.10_R10.0',
                  'RIGID5_dl0.10_R10.0 (gaussian_infl)',
                  'test',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


def SemanticKitti_multiframe():
    """
    Vote validation is not very good because votes are too slow... The validation set is not event entirely tested
    after 500 epochs!

    Either find a way to get faster validation or reduce validation set. We need to cover the whole validation in
    20-30 epoch at least.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2019-10-24_16-44-38'
    end = 'Log_2019-11-24_16-44-38'
    logs = np.sort([join('results', l) for l in listdir('results') if start <= l <= end])

    # Give names to the logs (for legends)
    logs_names = ['DEFORM_dl0.10_R10.0',
                  'DEFORM_dl0.10_R10.0',
                  'DEFORM_dl0.10_R6.0',
                  'test',
                  'test']
    logs_names = np.array(logs_names[:len(logs)])


    # Remove useless
    useless = []
    if useless:
        logs = np.delete(logs, useless)
        logs_names = np.delete(logs_names, useless)



    return logs, logs_names


if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # My logs: choose the logs to show
    exp_id = 29

    if exp_id == 0:
        logs, logs_names = scannet_small_in_radius()
    elif exp_id == 1:
        logs, logs_names = scannet_full_rooms()
    elif exp_id == 2:
        logs, logs_names = ModelNet_deeper_nets()
    elif exp_id == 3:
        logs, logs_names = scannet_deeper_nets()
    elif exp_id == 4:
        logs, logs_names = npm3d()
    elif exp_id == 5:
        logs, logs_names = semantic3D()
    elif exp_id == 6:
        logs, logs_names = semantic3D_alternate_validation_set()
    elif exp_id == 7:
        logs, logs_names = semantic3D_color_augmentation()
    elif exp_id == 8:
        logs, logs_names = ModelNet40_small()
    elif exp_id == 9:
        logs, logs_names = ModelNet40_full()
    elif exp_id == 10:
        logs, logs_names = ModelNet40_deformable()
    elif exp_id == 11:
        logs, logs_names = ModelNet40_deformable_full()
    elif exp_id == 12:
        logs, logs_names = Semantic3D_deformable()
    elif exp_id == 13:
        logs, logs_names = Scannet_deformable()
    elif exp_id == 14:
        logs, logs_names = Scannet_restricted()
    elif exp_id == 15:
        logs, logs_names = Scannet_restricted_bis()
    elif exp_id == 16:
        logs, logs_names = S3DIS_deform()
    elif exp_id == 17:
        logs, logs_names = ShapeNetPart_deform()
    elif exp_id == 18:
        logs, logs_names = S3DIS_k_fold()
    elif exp_id == 19:
        logs, logs_names = S3DIS_k_fold_bis()
    elif exp_id == 20:
        logs, logs_names = NPM3D_deformable()
    elif exp_id == 21:
        logs, logs_names = Scannet_more()
    elif exp_id == 22:
        logs, logs_names = S3DIS_new()
    elif exp_id == 23:
        logs, logs_names = ModelNet40_pretrained()
    elif exp_id == 24:
        logs, logs_names = ShapeNetPart_pretrained()
    elif exp_id == 25:
        logs, logs_names = multi_comparison()
    elif exp_id == 26:
        logs, logs_names = SemanticKitti_first()
    elif exp_id == 27:
        logs, logs_names = ModelNet40_optimization()
    elif exp_id == 28:
        logs, logs_names = SemanticKitti_soloframe()
    elif exp_id == 29:
        logs, logs_names = SemanticKitti_multiframe()


    else:
        raise ValueError('Unvalid exp_id')

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # Check if there is a multi dataset
    multi = []
    multi_datasets = []
    for log in logs:
        if ('val_IoUs.txt' in [f for f in listdir(log)]) or ('val_confs.txt' in [f for f in listdir(log)]):
            multi += [False]
        elif np.any(['val_IoUs' in f for f in listdir(log)]):
            multi += [True]
            config = MultiConfig()
            config.load(log)
            multi_datasets = np.unique(np.hstack((config.datasets, multi_datasets)))
        else:
            multi += [False]

    # Check that multi datasets are supported
    for name in multi_datasets:
        if name not in ['S3DIS', 'Scannet', 'Semantic3D', 'NPM3D']:
            raise ValueError('Unsupported dataset : ' + name)

    # Check single datasetes are in the multidatasets
    if np.any(multi):
        for log, is_multi in zip(logs, multi):
            if not is_multi:
                config = Config()
                config.load(log)
                if config.dataset not in multi_datasets:
                    raise ValueError('A single dataset is not in the multi dataset : ' + config.dataset)

        # Plot the training loss and accuracy
        compare_trainings(logs[multi], logs_names[multi])

        # Plot the validation
        compare_convergences_multicloud(logs, multi, multi_datasets, logs_names)

    else:

        # Check that all logs are of the same dataset. Different object can be compared
        plot_dataset = None
        for log in logs:
            config = Config()
            config.load(log)
            if 'ShapeNetPart' in config.dataset:
                this_dataset = 'ShapeNetPart'
            else:
                this_dataset = config.dataset
            if plot_dataset:
                if plot_dataset == this_dataset:
                    continue
                else:
                    raise ValueError('All logs must share the same dataset to be compared')
            else:
                plot_dataset = this_dataset

        # Plot the training loss and accuracy
        compare_trainings(logs, logs_names)

        # Plot the validation
        if plot_dataset.startswith('Shape'):
            compare_convergences_multisegment(logs, logs_names)
        elif plot_dataset.startswith('S3DIS'):
            dataset = S3DISDataset()
            compare_convergences_segment(dataset, logs, logs_names)
        elif plot_dataset.startswith('Model'):
            dataset = ModelNet40Dataset()
            compare_convergences_classif(dataset, logs, logs_names)
        elif plot_dataset.startswith('Scann'):
            dataset = ScannetDataset()
            compare_convergences_segment(dataset, logs, logs_names)
        elif plot_dataset.startswith('Semantic3D'):
            dataset = Semantic3DDataset()
            compare_convergences_segment(dataset, logs, logs_names)
        elif plot_dataset.startswith('NPM3D'):
            dataset = NPM3DDataset()
            compare_convergences_segment(dataset, logs, logs_names)
        elif plot_dataset.startswith('SemanticKitti'):
            dataset = SemanticKittiDataset()
            compare_convergences_SLAM(dataset, logs, logs_names)
        else:
            raise ValueError('Unsupported dataset : ' + plot_dataset)




