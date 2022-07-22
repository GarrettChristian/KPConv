#
#
#      0=========================0
#      |    Kernel Point CNN     |
#      0=========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Handle multiple dataset training in a class
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
import tensorflow as tf
import numpy as np
import time
import pickle
from sklearn.neighbors import KDTree
import os

# PLY reader
from utils.ply import read_ply, write_ply

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

# Dataset parent class
from datasets.common import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

from datasets.S3DIS import S3DISDataset
from datasets.Scannet import ScannetDataset
from datasets.Semantic3D import Semantic3DDataset
from datasets.NPM3D import NPM3DDataset


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


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class MultiDataset(Dataset):
    """
    Class to handle multiple dataset for segmentation task.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, dataset_names, input_threads=8):
        """
        Initiation method. Give the name of the datasets you want for training and the scalings used. For each dataset,
        the network will have a specific output at each scale? Or a single ouptu for all scales?
        :param dataset_names: list of string, the dataset names
        :param dataset_scales: list of list of floats. The scales for each dataset
        :param input_threads: NUmber of parallel threads for input computations
        """

        # Create a custom name for the dataset depending on which dataset is used
        names = [name[:5] for name in dataset_names]
        Dataset.__init__(self, 'Multi' + '_'.join(names))
        self.ignored_labels = np.array([])

        # Initiate the dataset classes
        self.datasets = []
        for name in dataset_names:
            if name.startswith('S3DIS'):
                self.datasets += [S3DISDataset(input_threads=input_threads)]
            elif name.startswith('Scann'):
                self.datasets += [ScannetDataset(input_threads=input_threads)]
            elif name.startswith('Seman'):
                self.datasets += [Semantic3DDataset(input_threads=input_threads)]
            elif name.startswith('NPM3D'):
                self.datasets += [NPM3DDataset(input_threads=input_threads)]
            else:
                raise ValueError('Unsupported dataset : ' + name)

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        self.network_model = 'multi_cloud_segmentation'

        # Number of input threads
        self.num_threads = input_threads

    def load_subsampled_clouds(self, scales):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        # For each dataset, load the subsampled clouds of the smallest scale
        for dataset, dataset_scales in zip(self.datasets, scales):

            print('\n')
            print(dataset.name)
            print('*'*len(dataset.name)+'\n')

            # Ensure the validation scale is in min and max values
            if not (dataset_scales[0] <= dataset_scales[-1] <= dataset_scales[1]):
                raise ValueError('Unvalid scales for dataset : ' + dataset.name)

            # Load the subsampled cloud special for multidataset training
            dataset.load_multi_clouds(dataset_scales[0], dataset_scales[-1])

        return

    def init_input_pipeline(self, config):
        """
        Prepare the input pipeline with tf.Dataset class
        """

        ######################
        # Calibrate parameters
        ######################

        print('Initiating input pipelines')

        # Update num classes in config
        config.num_classes = [dataset.num_classes - len(dataset.ignored_labels) for dataset in self.datasets]
        config.ignored_label_inds = [[d.label_to_idx[ign_l] for ign_l in d.ignored_labels] for d in self.datasets]

        # Update network model in config
        config.network_model = self.network_model

        # Calibrate generators to batch_num
        self.batch_limit = self.calibrate_batches(config)

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = np.full(config.num_layers, hist_n, dtype=np.int32)

        # Calibrate max neighbors number
        self.calibrate_neighbors(config)

        ################################
        # Initiate tensorflow parameters
        ################################

        # Reset graph
        tf.reset_default_graph()

        # Set random seed (You also have to set it in network_architectures.weight_variable)
        #np.random.seed(42)
        #tf.set_random_seed(42)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training', config)
        gen_function_val, _, _ = self.get_batch_gen('validation', config)
        map_func = self.get_tf_mapping(config)

        ##################
        # Training dataset
        ##################

        # Create batched dataset from generator
        self.train_data = tf.data.Dataset.from_generator(gen_function,
                                                         gen_types,
                                                         gen_shapes)

        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        self.train_data = self.train_data.prefetch(10)

        ##############
        # Test dataset
        ##############

        # Create batched dataset from generator
        self.val_data = tf.data.Dataset.from_generator(gen_function_val,
                                                       gen_types,
                                                       gen_shapes)

        # Transform inputs
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        self.val_data = self.val_data.prefetch(10)

        #################
        # Common iterator
        #################

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        self.flat_inputs = iter.get_next()

        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)


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

        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.epoch_steps * config.batch_num
            random_pick_n = None

        elif split == 'validation':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        elif split == 'test':

            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = config.validation_size * config.batch_num

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        if not hasattr(self, 'min_dataset_potentials'):
            self.min_dataset_potentials = {}
        self.min_dataset_potentials[split] = []

        for dataset in self.datasets:

            # Initiate potentials for regular generation
            if not hasattr(dataset, 'potentials'):
                dataset.potentials = {}
                dataset.min_potentials = {}

            # Reset potentials
            dataset.potentials[split] = []
            dataset.min_potentials[split] = []
            for i, tree in enumerate(dataset.input_colors[split]):
                dataset.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
                dataset.min_potentials[split] += [float(np.min(dataset.potentials[split][-1]))]

            # Compute min dataset potentials
            self.min_dataset_potentials[split] += [np.min(dataset.min_potentials[split])]

        ##########################
        # Def generators functions
        ##########################

        def spatially_regular_val_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []
            di_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):


                # Choose a random dataset
                d_i = int(np.argmin(self.min_dataset_potentials[split]))

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.datasets[d_i].min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.datasets[d_i].potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.datasets[d_i].input_trees[split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Choose a subsampling dl and radius of validation
                val_dl = config.scales[d_i][-1]
                val_R = config.input_scale_ratio * val_dl

                # Add noise to the center point
                noise = np.random.normal(scale=val_R/10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = self.datasets[d_i].input_trees[split][cloud_ind].query_radius(pick_point,
                                                                                           r=val_R)[0]

                # Update potentials (Tuckey weights)
                Tuckey_R = val_R
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / np.square(Tuckey_R))
                tukeys[dists > np.square(Tuckey_R)] = 0
                self.datasets[d_i].potentials[split][cloud_ind][input_inds] += tukeys
                self.datasets[d_i].min_potentials[split][cloud_ind] = float(
                    np.min(self.datasets[d_i].potentials[split][cloud_ind]))
                self.min_dataset_potentials[split][d_i] = np.min(self.datasets[d_i].min_potentials[split])

                # Number collected
                n = input_inds.shape[0]

                # Safe check for very dense areas
                if n > self.batch_limit:
                    input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                if hasattr(self.datasets[d_i], 'input_colors'):
                    if self.datasets[d_i].input_colors[split][cloud_ind].shape[1] == 3:
                        input_colors = self.datasets[d_i].input_colors[split][cloud_ind][input_inds]
                    else:
                        input_colors = input_points * 0
                else:
                    input_colors = input_points * 0
                input_labels = self.datasets[d_i].input_labels[split][cloud_ind][input_inds]
                input_labels = np.array([self.datasets[d_i].label_to_idx[l] for l in input_labels])

                # Rescale points for the network
                input_points *= config.in_radius / val_R

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32),
                           np.array(di_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    di_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]
                    di_list += [d_i]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32),
                       np.array(di_list, dtype=np.int32))

        def spatially_regular_multi_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []
            di_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):

                # Choose a random dataset
                d_i = int(np.argmin(self.min_dataset_potentials[split]))

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.datasets[d_i].min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.datasets[d_i].potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.datasets[d_i].input_trees[split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Choose a random subsampling dl, and thus a random radius for this input sphere
                random_dl = np.random.uniform(low=config.scales[d_i][0], high=config.scales[d_i][1])
                random_R = config.input_scale_ratio * random_dl

                # Add noise to the center point
                noise = np.random.normal(scale=random_R/10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = self.datasets[d_i].input_trees[split][cloud_ind].query_radius(pick_point,
                                                                                           r=random_R)[0]

                # Update potentials (Tuckey weights)
                Tuckey_R = config.input_scale_ratio * config.scales[d_i][0]
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / np.square(Tuckey_R))
                tukeys[dists > np.square(Tuckey_R)] = 0
                self.datasets[d_i].potentials[split][cloud_ind][input_inds] += tukeys
                self.datasets[d_i].min_potentials[split][cloud_ind] = float(
                    np.min(self.datasets[d_i].potentials[split][cloud_ind]))
                self.min_dataset_potentials[split][d_i] = np.min(self.datasets[d_i].min_potentials[split])

                # Get the points inside input sphere and subsample them
                neighbors = (points[input_inds] - pick_point).astype(np.float32)
                if hasattr(self.datasets[d_i], 'input_colors'):
                    if self.datasets[d_i].input_colors[split][cloud_ind].shape[1] == 3:
                        colors = self.datasets[d_i].input_colors[split][cloud_ind][input_inds]
                    else:
                        colors = neighbors * 0
                else:
                    colors = neighbors * 0
                labels = self.datasets[d_i].input_labels[split][cloud_ind][input_inds]
                sub_points, sub_colors, sub_labels = grid_subsampling(neighbors,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=random_dl)
                sub_labels = np.ravel(sub_labels)

                # Rescale points for the network
                sub_points *= config.in_radius / random_R

                # Number collected
                n = sub_points.shape[0]

                # Safe check for very dense areas
                if n > self.batch_limit:
                    rand_inds = np.random.choice(n, size=int(self.batch_limit)-1, replace=False)
                    sub_points = sub_points[rand_inds]
                    sub_colors = sub_colors[rand_inds]
                    sub_labels = sub_labels[rand_inds]
                    n = sub_points.shape[0]
                else:
                    rand_inds = np.arange(n)

                # Correct the labels
                sub_labels = np.array([self.datasets[d_i].label_to_idx[l] for l in sub_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32),
                           np.array(di_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    di_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [sub_points]
                    c_list += [np.hstack((sub_colors, sub_points + pick_point))]
                    pl_list += [sub_labels]
                    pi_list += [rand_inds]
                    ci_list += [cloud_ind]
                    di_list += [d_i]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32),
                       np.array(di_list, dtype=np.int32))

        ###################
        # Choose generators
        ###################

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_multi_gen

        elif split == 'validation':
            gen_func = spatially_regular_val_gen

        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds, dataset_inds):
            """
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]

            # Augmentation : randomly drop colors
            if config.in_features_dim in [4, 5]:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 2:
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
            elif config.in_features_dim == 3:
                stacked_features = stacked_colors
            elif config.in_features_dim == 4:
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif config.in_features_dim == 5:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]), axis=1)
            elif config.in_features_dim == 7:
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            input_list = self.tf_segmentation_inputs(config,
                                                     stacked_points,
                                                     stacked_features,
                                                     point_labels,
                                                     stacks_lengths,
                                                     batch_inds,
                                                     object_labels=dataset_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots]
            input_list += [point_inds, cloud_inds, dataset_inds]

            return input_list

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T

    def calibrate_batches(self, config):

        # Only training is supported
        split = 'training'

        # Number of points picked randomly accross the datasets
        cloud_num = np.sum([len(dataset.input_trees[split]) for dataset in self.datasets])
        N = (50000 // cloud_num) + 1
        sizes = []

        # Take a bunch of example neighborhoods in all clouds
        for dataset, dataset_scales in zip(self.datasets, config.scales):
            for tree in dataset.input_trees[split]:

                # Randomly pick points
                points = np.array(tree.data, copy=False)
                rand_inds = np.random.choice(points.shape[0], size=N, replace=False)
                neighbors = tree.query_radius(points[rand_inds], r=config.input_scale_ratio * dataset_scales[0])

                # Only save neighbors lengths
                sizes += [len(neighb) for neighb in neighbors]

        sizes = np.sort(sizes)

        # Higher bound for batch limit
        lim = sizes[-1] * config.batch_num

        # Biggest batch size with this limit
        sum_s = 0
        max_b = 0
        for i, s in enumerate(sizes):
            sum_s += s
            if sum_s > lim:
                max_b = i
                break

        # With a proportional corrector, find batch limit which gets the wanted batch_num
        estim_b = 0
        for i in range(10000):
            # Compute a random batch
            rand_shapes = np.random.choice(sizes, size=max_b, replace=False)
            b = np.sum(np.cumsum(rand_shapes) < lim)

            # Update estim_b (low pass filter istead of real mean
            estim_b += (b - estim_b) / min(i+1, 100)

            # Correct batch limit
            lim += 10.0 * (config.batch_num - estim_b)

        return lim

    def calibrate_neighbors(self, config, keep_ratio=0.8, samples_threshold=10000):

        # Create a tensorflow input pipeline
        # **********************************

        # Calibrate with training split
        split = 'training'

        # Get mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen(split, config)
        map_func = self.get_tf_mapping(config)

        # Create batched dataset from generator
        train_data = tf.data.Dataset.from_generator(gen_function,
                                                    gen_types,
                                                    gen_shapes)

        train_data = train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        train_data = train_data.prefetch(10)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        flat_inputs = iter.get_next()

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_data)

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Create a local session for the calibration.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        with tf.Session(config=cProto) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Initialise iterator with train data
            sess.run(train_init_op)

            # Get histogram of neighborhood sizes in 1 epoch max
            # **************************************************

            neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
            t0 = time.time()
            mean_dt = np.zeros(2)
            last_display = t0
            epoch = 0
            training_step = 0
            while epoch < 1 and np.min(np.sum(neighb_hists, axis=1)) < samples_threshold:
                try:

                    # Get next inputs
                    t = [time.time()]
                    ops = flat_inputs[config.num_layers:2 * config.num_layers]
                    neighbors = sess.run(ops)
                    t += [time.time()]

                    # Update histogram
                    counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)
                    t += [time.time()]

                    # Average timing
                    mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Console display
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'Calib Neighbors {:08d} : timings {:4.2f} {:4.2f}'
                        print(message.format(training_step,
                                             1000 * mean_dt[0],
                                             1000 * mean_dt[1]))

                    training_step += 1

                except tf.errors.OutOfRangeError:
                    print('End of train dataset')
                    epoch += 1

            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

            self.neighborhood_limits = percentiles
            print('\n')

        return



    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

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
                batches = np_flat_inputs[-8]
                n_b = 0.99 * n_b + 0.01 * batches.shape[0]
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

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