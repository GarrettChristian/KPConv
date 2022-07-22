#
#
#      0=========================0
#      |    Kernel Point CNN     |
#      0=========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Handle SemanticKitti dataset in a class
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

# Basic libs
import json
import os
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
import yaml
import sys

# PLY reader
from utils.ply import read_ply, write_ply
from utils.mesh import rasterize_mesh

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

# Dataset parent class
from datasets.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

#from mayavi import mlab

# Subsampling as a tf custom op
from datasets.common import tf_batch_subsampling

# Load custom operation
tf_batch_subsampling_features_module = tf.load_op_library('tf_custom_ops/tf_batch_subsampling_features.so')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)

def tf_batch_subsampling_features(points, features, labels, batches_len, sampleDl, max_points=0):
    return tf_batch_subsampling_features_module.batch_grid_subsampling_features(points,
                                                                                features,
                                                                                labels,
                                                                                batches_len,
                                                                                sampleDl,
                                                                                max_points)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class SemanticKittiDataset(Dataset):
    """
    Class to handle SemanticKitti dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, input_threads=8, n_frames=1):
        Dataset.__init__(self, 'SemanticKitti')

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        self.network_model = 'slam_segmentation'

        # Number of input threads
        self.num_threads = input_threads

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing ply files
        if 'SSH_CLIENT' in os.environ.keys():
            self.path = '/home/hugues/Data/SemanticKitti'
        else:
            self.path = '/media/hugues/Data/These/Datasets/SemanticKitti/dataset'

        # Get a list of sequences
        self.train_sequences = ['{:02d}'.format(i) for i in range(11)]
        self.test_sequences = ['{:02d}'.format(i) for i in range(11, 22)]
        self.seq_splits = [int(i == 8) for i in range(11)]
        self.validation_split = 1

        # List all files in each sequence
        self.train_frames = []
        for seq in self.train_sequences:
            velo_path = join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
            self.train_frames.append(frames)

        self.test_frames = []
        for seq in self.test_sequences:
            velo_path = join(self.path, 'sequences', seq, 'velodyne')
            frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
            self.test_frames.append(frames)

        self.frames = {}
        self.frames['training'] = self.train_frames
        self.frames['validation'] = self.train_frames
        self.frames['test'] = self.test_frames

        ###########################
        # Object classes parameters
        ###########################

        # Read labels
        if n_frames == 1:
            config_file = join(self.path, 'semantic-kitti.yaml')
        elif n_frames > 1:
            config_file = join(self.path, 'semantic-kitti-all.yaml')
        else:
            raise ValueError('number of frames has to be >= 1')

        with open(config_file, 'r') as stream:
            doc = yaml.safe_load(stream)
            all_labels = doc['labels']
            learning_map_inv = doc['learning_map_inv']
            learning_map = doc['learning_map']
            self.learning_map = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map.items():
                self.learning_map[k] = v

            self.learning_map_inv = np.zeros((np.max([k for k in learning_map_inv.keys()]) + 1), dtype=np.int32)
            for k, v in learning_map_inv.items():
                self.learning_map_inv[k] = v


        # Dict from labels to names
        self.label_to_names = {k: all_labels[v] for k, v in learning_map_inv.items()}

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.sort([0])

    def load_calib_poses(self, n_frames):
        """
        load calib poses and times.
        """

        #
        #   QUESTION : la reprojection des indices pour le validation score? trop de points donc
        #

        ########
        # Train
        ########

        self.train_calibrations = []
        self.train_times = []
        self.train_poses = []

        for seq in self.train_sequences:

            seq_folder = join(self.path, 'sequences', seq)

            # Read Calib
            self.train_calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.train_times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.train_calibrations[-1])
            self.train_poses.append([pose.astype(np.float32) for pose in poses_f64])

        #######
        # Test
        #######

        self.test_calibrations = []
        self.test_times = []
        self.test_poses = []

        for seq in self.test_sequences:

            seq_folder = join(self.path, 'sequences', seq)

            # Read Calib
            self.test_calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.test_times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.test_calibrations[-1])
            self.test_poses.append([pose.astype(np.float32) for pose in poses_f64])

        ###################################
        # Prepare the indices of all frames
        ###################################

        self.all_inds = {}

        # Training
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.train_frames)
                              if self.seq_splits[i] != self.validation_split])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for i, _ in enumerate(self.train_frames)
                                if self.seq_splits[i] != self.validation_split])
        self.all_inds['training'] = np.vstack((seq_inds, frame_inds)).T

        # Validation
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.train_frames)
                              if self.seq_splits[i] == self.validation_split])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for i, _ in enumerate(self.train_frames)
                                if self.seq_splits[i] == self.validation_split])
        self.all_inds['validation'] = np.vstack((seq_inds, frame_inds)).T

        # Test
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32) * i for i, _ in enumerate(self.test_frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.test_frames])
        self.all_inds['test'] = np.vstack((seq_inds, frame_inds)).T

        ################################################
        # For each class list the frames containing them
        ################################################

        class_frames_bool = {}
        class_frames_bool['training'] = np.zeros((0, self.num_classes), dtype=np.bool)
        class_frames_bool['validation'] = np.zeros((0, self.num_classes), dtype=np.bool)

        self.class_proportions = {}
        self.class_proportions['training'] = np.zeros((self.num_classes,), dtype=np.int32)
        self.class_proportions['validation'] = np.zeros((self.num_classes,), dtype=np.int32)

        for s_ind, (seq, seq_frames) in enumerate(zip(self.train_sequences, self.train_frames)):

            frame_mode = 'single'
            if n_frames > 1:
                frame_mode = 'multi'
            seq_stat_file = join(self.path, 'sequences', seq, 'stats_{:s}.pkl'.format(frame_mode))

            # Check if inputs have already been computed
            if isfile(seq_stat_file):
                # Read pkl
                with open(seq_stat_file, 'rb') as f:
                    seq_class_frames, seq_proportions = pickle.load(f)

            else:

                # Initiate dict
                print('Preparing seq {:s} class frames. (Long but one time only)'.format(seq))

                # Class frames as a boolean mask
                seq_class_frames = np.zeros((len(seq_frames), self.num_classes), dtype=np.bool)

                # Proportion of each class
                seq_proportions = np.zeros((self.num_classes,), dtype=np.int32)

                # Sequence path
                seq_path = join(self.path, 'sequences', seq)

                # Read all frames
                for f_ind, frame_name in enumerate(seq_frames):

                    # Path of points and labels
                    label_file = join(seq_path, 'labels', frame_name + '.label')

                    # Read labels
                    frame_labels = np.fromfile(label_file, dtype=np.int32)
                    sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                    sem_labels = self.learning_map[sem_labels]

                    # Get present labels and there frequency
                    unique, counts = np.unique(sem_labels, return_counts=True)

                    # Add this frame to the frame lists of all class present
                    frame_labels = np.array([self.label_to_idx[l] for l in unique], dtype=np.int32)
                    seq_class_frames[f_ind, frame_labels] = True

                    # Add proportions
                    seq_proportions[frame_labels] += counts

                # Save pickle
                with open(seq_stat_file, 'wb') as f:
                    pickle.dump([seq_class_frames, seq_proportions], f)

            # Add the sequence class_frames to the split class_frames
            split = 'training'
            if self.seq_splits[s_ind] == self.validation_split:
                split = 'validation'

            class_frames_bool[split] = np.vstack((class_frames_bool[split], seq_class_frames))
            self.class_proportions[split] += seq_proportions

        # Transform boolean indexing to int indices.
        self.class_frames = {}
        self.class_frames['training'] = []
        self.class_frames['validation'] = []
        for split in ['training', 'validation']:
            for i, c in enumerate(self.label_values):
                if c in self.ignored_labels:
                    self.class_frames[split].append(np.zeros((0,), dtype=np.int32))
                else:
                    integer_inds = np.where(class_frames_bool[split][:, i])[0]
                    self.class_frames[split].append(integer_inds.astype(np.int32))

        ################################################
        # For each class list the frames containing them
        ################################################

        # Add variables for validation
        self.val_points = []
        self.val_labels = []
        self.val_confs = []

        for s_ind, seq_frames in enumerate(self.train_frames):

            if self.seq_splits[s_ind] == self.validation_split:
                self.val_confs.append(np.zeros((len(seq_frames), self.num_classes, self.num_classes)))
            else:
                self.val_confs.append(np.zeros((0,)))

        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
        if not hasattr(self, 'f_potentials'):
            self.f_potentials = {}

        # Initiate parameters depending on the chosen split
        epoch_n = None
        if split == 'training':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.epoch_steps * config.batch_num

        elif split == 'validation':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'test':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num


        # Reset potentials
        self.potentials[split] = np.random.rand(self.all_inds[split].shape[0]) * 0.1 + 0.1
        self.f_potentials[split] = []
        for i, seq_frames in enumerate(self.frames[split]):
            if split == 'test' or self.seq_splits[i] == self.validation_split:
                self.f_potentials[split].append([np.zeros((0,)) for _ in seq_frames])
            else:
                self.f_potentials[split].append([])

        ################
        # Def generators
        ################

        def random_gen():

            # Initiate concatenation lists
            tp_list = []
            tn_list = []
            tl_list = []
            ti_list = []
            tp0_list = []
            batch_n = 0

            # Get the list of indices to generate thanks to potentials
            if epoch_n < self.potentials[split].shape[0]:
                gen_indices = np.argpartition(self.potentials[split], epoch_n)[:epoch_n]
            else:
                gen_indices = np.random.permutation(self.potentials[split].shape[0])

            # Update potentials (Change the order for the next epoch)
            self.potentials[split][gen_indices] = np.ceil(self.potentials[split][gen_indices])
            self.potentials[split][gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

            # Generator loop
            for s_ind, f_ind in self.all_inds[split][gen_indices]:

                #########################
                # Merge n_frames together
                #########################

                # Initiate merged points
                merged_points = np.zeros((0, 4), dtype=np.float32)
                merged_labels = np.zeros((0,), dtype=np.int32)
                merged_coords = np.zeros((0, 4), dtype=np.float32)

                # Get center of the first frame in world coordinates
                p_origin = np.zeros((1, 4))
                p_origin[0, 3] = 1
                if split == 'test':
                    pose0 = self.test_poses[s_ind][f_ind]
                else:
                    pose0 = self.train_poses[s_ind][f_ind]
                p0 = p_origin.dot(pose0.T)[:, :3]
                p0 = np.squeeze(p0)

                num_merged = 0
                f_inc = 0
                while num_merged < config.n_frames and f_ind - f_inc >= 0:

                    # Select frame only if center has moved far away (more than 1 meter)
                    if split == 'test':
                        pose = self.test_poses[s_ind][f_ind - f_inc]
                    else:
                        pose = self.train_poses[s_ind][f_ind - f_inc]
                    diff = p_origin.dot(pose.T)[:, :3] - p_origin.dot(pose0.T)[:, :3]
                    if num_merged > 0 and np.linalg.norm(diff) < num_merged * -1.0:
                        f_inc += 1
                        continue

                    # Path of points and labels
                    if split == 'test':
                        seq_path = join(self.path, 'sequences', self.test_sequences[s_ind])
                        velo_file = join(seq_path, 'velodyne', self.test_frames[s_ind][f_ind - f_inc] + '.bin')
                        label_file = None
                    else:
                        seq_path = join(self.path, 'sequences', self.train_sequences[s_ind])
                        velo_file = join(seq_path, 'velodyne', self.train_frames[s_ind][f_ind - f_inc] + '.bin')
                        label_file = join(seq_path, 'labels', self.train_frames[s_ind][f_ind - f_inc] + '.label')

                    # Read points
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    points = frame_points.reshape((-1, 4))

                    if split == 'test':
                        # Fake labels
                        sem_labels = np.zeros((points.shape[0],), dtype=np.int32)
                    else:
                        # Read labels
                        frame_labels = np.fromfile(label_file, dtype=np.int32)
                        sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                        sem_labels = self.learning_map[sem_labels]

                    # Apply pose
                    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
                    new_points = hpoints.dot(pose.T)
                    new_points[:, 3:] = points[:, 3:]

                    # In case of validation, keep the original points in memory
                    if split in ['validation', 'test'] and f_inc == 0:
                        self.val_points.append(new_points[:, :3])
                        self.val_labels.append(sem_labels)

                    # In case radius smaller than 50m, choose the center of the input sphere with potential
                    if config.in_radius < 50.0 and f_inc == 0:

                        # Initiate potentials if not done
                        if self.f_potentials[split][s_ind][f_ind].shape[0] == 0:
                            self.f_potentials[split][s_ind][f_ind] = np.random.rand(sem_labels.shape[0])\
                                                                         .astype(np.float32) * 0.01
                            mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) > 50 ** 2
                            self.f_potentials[split][s_ind][f_ind][mask] = 1000.0

                        # Choose center as potential minimum
                        p0_ind = np.argmin(self.f_potentials[split][s_ind][f_ind])
                        p0 = new_points[p0_ind, :3]

                        # Update frame potentials
                        dists = np.sum(np.square((new_points[:, :3] - p0).astype(np.float32)), axis=1)
                        tukeys = np.square(1 - dists / np.square(config.in_radius))
                        tukeys[dists > np.square(config.in_radius)] = 0
                        self.f_potentials[split][s_ind][f_ind] += tukeys

                    # Eliminate points further than config.in_radius
                    mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < config.in_radius**2
                    mask_inds = np.where(mask)[0].astype(np.int32)

                    # Shuffle points
                    rand_order = np.random.permutation(mask_inds)
                    #rand_order = np.random.permutation(new_points.shape[0])
                    new_points = new_points[rand_order, :]
                    sem_labels = sem_labels[rand_order]

                    # Place points in original frame reference to get coordinates
                    hpoints = np.hstack((new_points[:, :3], np.ones_like(new_points[:, :1])))
                    new_coords = hpoints.dot(pose0)
                    new_coords[:, 3:] = new_points[:, 3:]

                    # Increment merge count
                    merged_points = np.vstack((merged_points, new_points))
                    merged_labels = np.hstack((merged_labels, sem_labels))
                    merged_coords = np.vstack((merged_coords, new_coords))
                    num_merged += 1
                    f_inc += 1

                # Too see yielding speed with debug timings method, collapse points (reduce mapping time to nearly 0)
                #merged_points = merged_points[:100, :]
                #merged_labels = merged_labels[:100]
                #merged_points *= 0.1

                # In case batch is full, yield it and reset it
                if batch_n >= config.batch_num:
                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tn_list, axis=0),
                           np.concatenate(tl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.stack(tp0_list, axis=0),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tn_list = []
                    tl_list = []
                    ti_list = []
                    tp0_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [merged_points[:, :3]]
                tn_list += [merged_coords]
                tl_list += [merged_labels]
                ti_list += [[s_ind, f_ind]]
                tp0_list += [p0]

                # Update batch size
                batch_n += 1

            yield (np.concatenate(tp_list, axis=0),
                   np.concatenate(tn_list, axis=0),
                   np.concatenate(tl_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.stack(tp0_list, axis=0),
                   np.array([tp.shape[0] for tp in tp_list]))

        def balanced_class_gen():

            # Initiate concatenation lists
            tp_list = []
            tn_list = []
            tl_list = []
            ti_list = []
            tp0_list = []
            batch_n = 0

            # Generate a list of indices balancing classes and respecting potentials
            gen_indices = []
            gen_classes = []

            for i, c in enumerate(self.label_values):

                if c not in self.ignored_labels:

                    # Get the potentials of the frames containing this class
                    class_potentials = self.potentials[split][self.class_frames[split][i]]

                    # Get the indices to generate thanks to potentials
                    class_n = epoch_n // self.num_classes + 1
                    if class_n < class_potentials.shape[0]:
                        class_indices = np.argpartition(class_potentials, class_n)[:class_n]
                    else:
                        class_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = self.class_frames[split][i][class_indices]

                    # Add the indices to the generated ones
                    gen_indices.append(class_indices)
                    gen_classes.append(class_indices * 0 + c)

                    # Update potentials
                    self.potentials[split][class_indices] = np.ceil(self.potentials[split][class_indices])
                    self.potentials[split][class_indices] += np.random.rand(class_indices.shape[0]) * 0.1 + 0.1

            # Stack the chosen indices of all classes
            gen_indices = np.hstack(gen_indices)
            gen_classes = np.hstack(gen_classes)

            # Shuffle generated indices
            rand_order = np.random.permutation(gen_indices.shape[0])
            gen_indices = gen_indices[rand_order]
            gen_classes = gen_classes[rand_order]

            # Generator loop
            for (s_ind, f_ind), wanted_label in zip(self.all_inds[split][gen_indices], gen_classes):

                #########################
                # Merge n_frames together
                #########################

                # Initiate merged points
                merged_points = np.zeros((0, 4), dtype=np.float32)
                merged_labels = np.zeros((0,), dtype=np.int32)
                merged_coords = np.zeros((0, 4), dtype=np.float32)

                # Get center of the first frame in world coordinates
                p_origin = np.zeros((1, 4))
                p_origin[0, 3] = 1
                if split == 'test':
                    pose0 = self.test_poses[s_ind][f_ind]
                else:
                    pose0 = self.train_poses[s_ind][f_ind]
                p0 = p_origin.dot(pose0.T)[:, :3]
                p0 = np.squeeze(p0)

                num_merged = 0
                f_inc = 0
                while num_merged < config.n_frames and f_ind - f_inc >= 0:

                    # Select frame only if center has moved far away (more than 1 meter)
                    pose = self.train_poses[s_ind][f_ind - f_inc]
                    diff = p_origin.dot(pose.T)[:, :3] - p_origin.dot(pose0.T)[:, :3]
                    if num_merged > 0 and np.linalg.norm(diff) < num_merged * -1.0:
                        f_inc += 1
                        continue

                    # Path of points and labels
                    if split == 'test':
                        seq_path = join(self.path, 'sequences', self.test_sequences[s_ind])
                        velo_file = join(seq_path, 'velodyne', self.test_frames[s_ind][f_ind - f_inc] + '.bin')
                        label_file = None
                    else:
                        seq_path = join(self.path, 'sequences', self.train_sequences[s_ind])
                        velo_file = join(seq_path, 'velodyne', self.train_frames[s_ind][f_ind - f_inc] + '.bin')
                        label_file = join(seq_path, 'labels', self.train_frames[s_ind][f_ind - f_inc] + '.label')

                        # Read points
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    points = frame_points.reshape((-1, 4))

                    if split == 'test':
                        # Fake labels
                        sem_labels = np.zeros((frame_points.shape[0],), dtype=np.int32)
                    else:
                        # Read labels
                        frame_labels = np.fromfile(label_file, dtype=np.int32)
                        sem_labels = frame_labels & 0xFFFF  # semantic label in lower half
                        sem_labels = self.learning_map[sem_labels]

                    # Apply pose
                    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
                    new_points = hpoints.dot(pose.T)
                    new_points[:, 3:] = points[:, 3:]

                    # In case of validation, keep the original points in memory
                    if split in ['validation', 'test'] and f_inc == 0:
                        self.val_points.append(new_points[:, :3])
                        self.val_labels.append(sem_labels)

                    # In case radius smaller than 50m, chose center on a point of the wanted class
                    if config.in_radius < 50.0 and f_inc == 0:
                        wanted_ind = np.random.choice(np.where(sem_labels == wanted_label)[0])
                        p0 = new_points[wanted_ind, :3]

                    # Eliminate points further than config.in_radius
                    mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < config.in_radius**2
                    mask_inds = np.where(mask)[0].astype(np.int32)

                    # Shuffle points
                    rand_order = np.random.permutation(mask_inds)
                    new_points = new_points[rand_order, :]
                    sem_labels = sem_labels[rand_order]

                    # Place points in original frame reference to get coordinates
                    hpoints = np.hstack((new_points[:, :3], np.ones_like(new_points[:, :1])))
                    new_coords = hpoints.dot(pose0)
                    new_coords[:, 3:] = new_points[:, 3:]

                    # Increment merge count
                    merged_points = np.vstack((merged_points, new_points))
                    merged_labels = np.hstack((merged_labels, sem_labels))
                    merged_coords = np.vstack((merged_coords, new_coords))
                    num_merged += 1
                    f_inc += 1

                # Too see yielding speed with debug timings method, collapse points (reduce mapping time to nearly 0)
                #merged_points = merged_points[:100, :]
                #merged_labels = merged_labels[:100]
                #merged_points *= 0.1

                # In case batch is full, yield it and reset it
                if batch_n >= config.batch_num:
                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tn_list, axis=0),
                           np.concatenate(tl_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.stack(tp0_list, axis=0),
                           np.array([tp.shape[0] for tp in tp_list]))
                    tp_list = []
                    tn_list = []
                    tl_list = []
                    ti_list = []
                    tp0_list = []
                    batch_n = 0

                # Add data to current batch
                tp_list += [merged_points[:, :3]]
                tn_list += [merged_coords]
                tl_list += [merged_labels]
                ti_list += [[s_ind, f_ind]]
                tp0_list += [p0]

                # Update batch size
                batch_n += 1

            yield (np.concatenate(tp_list, axis=0),
                   np.concatenate(tn_list, axis=0),
                   np.concatenate(tl_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.stack(tp0_list, axis=0),
                   np.array([tp.shape[0] for tp in tp_list]))

            return

        ##################
        # Return generator
        ##################

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = balanced_class_gen

        elif split == 'validation':
            gen_func = random_gen

        elif split == 'test':
            gen_func = random_gen

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Generator types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, 4], [None], [None, 2], [None, 3], [None])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, frame_inds, frame_centers, stacks_lengths):
            """
            """

            # Subsample input cloud
            s_points, s_colors, s_labels, s_l = tf_batch_subsampling_features(stacked_points,
                                                                              stacked_colors,
                                                                              point_labels,
                                                                              stacks_lengths,
                                                                              sampleDl=config.first_subsampling_dl,
                                                                              max_points=config.max_in_points)

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(s_l)

            # Augment input points
            s_points, scales, rots = self.tf_augment_input(s_points,
                                                           batch_inds,
                                                           config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            s_features = tf.ones((tf.shape(s_points)[0], 1), dtype=tf.float32)

            # Augmentation : randomly drop colors
            if config.in_features_dim in [2]:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                s_colors = s_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 2:
                s_features = tf.concat((s_features, s_colors[:, 3:]), axis=1)
            elif config.in_features_dim == 4:
                s_features = tf.concat((s_features, s_colors[:, :3]), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 2 ')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     s_points,
                                                     s_features,
                                                     s_labels,
                                                     s_l,
                                                     batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots]
            input_list += [frame_inds, frame_centers]

            return input_list

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Read points
        frame_points = np.fromfile(file_path, dtype=np.float32)
        points = frame_points.reshape((-1, 4))

        # Apply pose
        pose = self.train_poses[s_ind][f_ind]
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = hpoints.dot(pose.T)

        return new_points[:, :3]

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def check_IO_speed(self):

        # max tests
        test_iter = 10000

        # Prepare an epoch (all frames taken by the network in a random order
        seq_inds = np.hstack([np.ones(len(_), dtype=np.int32)*i for i, _ in enumerate(self.train_frames)])
        frame_inds = np.hstack([np.arange(len(_), dtype=np.int32) for _ in self.train_frames])
        all_inds = np.vstack((seq_inds, frame_inds)).T
        epoch_order = np.random.permutation(all_inds.shape[0])

        # timing variable
        mean_dt = np.zeros(1)
        last_display = time.time()

        for iter, (s_ind, f_ind) in enumerate(all_inds[epoch_order]):

            #print('reading the frame {:d} from sequence {:02d}'.format(f_ind, s_ind))

            t = [time.time()]

            # Path of points and labels
            seq_path = join(self.path, 'sequences', self.train_sequences[s_ind])
            velo_file = join(seq_path, 'velodyne', self.train_frames[s_ind][f_ind] + '.bin')
            label_file = join(seq_path, 'labels', self.train_frames[s_ind][f_ind] + '.label')

            # Read points
            frame_points = np.fromfile(velo_file, dtype=np.float32)
            points = frame_points.reshape((-1, 4))

            t += [time.time()]

            # Read labels
            frame_labels = np.fromfile(label_file, dtype=np.int32)
            sem_labels = frame_labels & 0xFFFF   # semantic label in lower half
            sem_labels = self.learning_map[sem_labels]

            t += [time.time()]

            # Apply pose
            pose = self.train_poses[s_ind][f_ind]
            hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
            new_points = hpoints.dot(pose.T)
            new_points[:, 3:] = points[:, 3:]

            # Eliminate points further than config.in_radius
            p0 = np.zeros((1, 4))
            p0[0, 3] = 1
            p0 = p0.dot(pose.T)[:, :3]
            mask = np.sum(np.square(new_points[:, :3] - p0), axis=1) < 51.0**2
            new_points = new_points[mask, :]

            t += [time.time()]

            # Average timing
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:05d} : timings {:4.2f} + {:4.2f} + {:4.2f} = {:4.2f}'
                print(message.format(iter,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     1000 * mean_dt[2],
                                     1000 * np.sum(mean_dt)), np.unique(sem_labels))

            # Do not do to many iterations
            if iter > test_iter:
                break

        return

    def check_input_pipeline_timing(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        n_b = config.batch_num
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-6]
                n_b = 0.99 * n_b + 0.01 * batches.shape[0]
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                debug_frame = False
                if debug_frame:
                    frame_inds = np_flat_inputs[-1][0]
                    labels = np_flat_inputs[-4]
                    features = np_flat_inputs[-8]
                    write_ply('test_{:02d}_{:d}.ply'.format(frame_inds[0], frame_inds[1]),
                              [points[0], features[:, 0:], labels],
                              ['x', 'y', 'z', 'testtt', 'cx', 'cy', 'cz', 'class'])
                    a = 1/0

                # Console display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d} => b = {:.1f}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1],
                                         n_b))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_batches(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        mean_b = 0
        min_b = 1000000
        max_b = 0
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                max_ind = np.max(batches)
                batches_len = [np.sum(b < max_ind-0.5) for b in batches]

                for b_l in batches_len:
                    mean_b = 0.99 * mean_b + 0.01 * b_l
                max_b = max(max_b, np.max(batches_len))
                min_b = min(min_b, np.min(batches_len))

                print('{:d} < {:.1f} < {:d} /'.format(min_b, mean_b, max_b),
                      self.training_batch_limit,
                      batches_len)

                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_neighbors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        hist_n = 500
        neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                for neighb_mat in neighbors:
                    print(neighb_mat.shape)

                counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                hists = [np.bincount(c, minlength=hist_n) for c in counts]

                neighb_hists += np.vstack(hists)

                print('***********************')
                dispstr = ''
                fmt_l = len(str(int(np.max(neighb_hists)))) + 1
                for neighb_hist in neighb_hists:
                    for v in neighb_hist:
                        dispstr += '{num:{fill}{width}}'.format(num=v, fill=' ', width=fmt_l)
                    dispstr += '\n'
                print(dispstr)
                print('***********************')

                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_colors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        t0 = time.time()
        mean_dt = np.zeros(2)
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                stacked_points = np_flat_inputs[:config.num_layers]
                stacked_colors = np_flat_inputs[-9]
                batches = np_flat_inputs[-7]
                stacked_labels = np_flat_inputs[-5]

                # Extract a point cloud and its color to save
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get points and colors (only for the concerned parts)
                    points = stacked_points[0][b]
                    colors = stacked_colors[b]
                    labels = stacked_labels[b]

                    write_ply('S3DIS_input_{:d}.ply'.format(b_i),
                              [points, colors[:, 1:4], labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'labels'])

                a = 1/0



                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_debug_input(self, config, path):

        # Get debug file
        file = join(path, 'all_debug_inputs.pkl')
        with open(file, 'rb') as f1:
            inputs = pickle.load(f1)

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) +1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / (np.prod(upsamples.shape) +1e-6)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))

            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))


        print('\nFinished\n\n')

    def check_input_potentials(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.val_init_op)

        # Initiate variables for runing model
        pots = []
        pot_points = None
        mean_dt = np.zeros(2)
        last_display = time.time()
        val_i = 0
        in_radius = None
        for i0 in range(config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Restructure flatten inputs
                s_points = np_flat_inputs[0]
                batches = np_flat_inputs[-7]
                S = np_flat_inputs[-4]
                R = np_flat_inputs[-3]
                f_inds = np_flat_inputs[-2]
                p0s = np_flat_inputs[-1]

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get prediction (only for the concerned parts)
                    points = s_points[b, :]
                    S_i = S[b_i]
                    R_i = R[b_i]
                    p0 = p0s[b_i]

                    # Get input points in their original positions
                    #points2 = (points * (1/S_i)).dot(R_i.T)

                    # get val_points that are in range
                    f_points = self.val_points[val_i]

                    if val_i == 0:

                        in_radius = self.f_potentials['validation'][f_inds[b_i, 0]][f_inds[b_i, 1]] < 10.0
                        pot_points = f_points[in_radius, :]

                    # Save points and pots
                    pots += [np.copy(self.f_potentials['validation'][f_inds[b_i, 0]][f_inds[b_i, 1]][in_radius])]

                    print(np.mean(pots[0]), np.mean(pots[-1]))


                    val_i += 1

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                    print(message.format(100 * i0 / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1])))

            except tf.errors.OutOfRangeError:
                break

        # Create figure for features
        fig1 = mlab.figure('Models', bgcolor=(1, 1, 1), size=(1000, 800))
        fig1.scene.parallel_projection = False

        # Indices
        global pots_i, activations
        pots_i = 0
        activations = None

        def initiate_scene():
            global activations

            #  clear figure
            mlab.clf(fig1)

            # Show point clouds colorized with activations
            activations = mlab.points3d(pot_points[:, 0],
                                        pot_points[:, 1],
                                        pot_points[:, 2],
                                        pots[pots_i],
                                        scale_factor=0.5,
                                        scale_mode='none',
                                        figure=fig1)

            # New title
            mlab.title(str(pots_i), color=(0, 0, 0), size=0.3, height=0.01)
            text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
            mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
            mlab.orientation_axes()

            return

        def update_scene():
            global activations

            # Show point clouds colorized with activations
            print(pots_i, np.mean(pots[pots_i]))
            activations.mlab_source.scalars = pots[pots_i]

            # New title
            mlab.title(str(pots_i), color=(0, 0, 0), size=0.3, height=0.01)
            text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
            mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
            mlab.orientation_axes()

            return

        def keyboard_callback(vtk_obj, event):
            global pots_i

            if vtk_obj.GetKeyCode() in ['g', 'G']:

                pots_i = (pots_i - 1) % len(pots)
                update_scene()

            elif vtk_obj.GetKeyCode() in ['h', 'H']:

                pots_i = (pots_i + 1) % len(pots)
                update_scene()

            return

        # Draw a first plot
        initiate_scene()
        fig1.scene.interactor.add_observer('KeyPressEvent', keyboard_callback)
        mlab.show()

        return







