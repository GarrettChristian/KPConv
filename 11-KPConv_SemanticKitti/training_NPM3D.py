#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on NPM3D dataset
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
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

# Custom libs
from utils.config import Config
from utils.trainer import ModelTrainer
from models.KPFCNN_model import KernelPointFCNN

# Dataset
from datasets.NPM3D import NPM3DDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


class NPM3DConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name in the format 'ShapeNetPart_Object' to segment an object class independently or 'ShapeNetPart_multi'
    # to segment all objects with a single model.
    dataset = 'NPM3D'

    # Number of classes in the dataset (This value is overwritten by dataset class when initiating input pipeline).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    network_model = None

    # Number of CPU threads for the input pipeline
    input_threads = 8

    #########################
    # Architecture definition
    #########################

    # Define layers
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'resnetb_deformable_strided',
                    'resnetb_deformable',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    # KPConv specific parameters
    num_kernel_points = 15
    first_subsampling_dl = 0.08
    in_radius = 4.0

    # Density of neighborhoods for deformable convs (which need bigger radiuses). For normal conv we use KP_extent
    density_parameter = 5.0

    # Behavior of convolutions in ('constant', 'linear', gaussian)
    KP_influence = 'linear'
    KP_extent = 1.0

    # Behavior of convolutions in ('closest', 'sum')
    convolution_mode = 'sum'

    # Can the network learn modulations
    modulated = False

    # Offset loss
    # 'permissive' only constrains offsets inside the big radius
    # 'fitting' helps deformed kernels to adapt to the geometry by penalizing distance to input points
    offsets_loss = 'fitting'
    offsets_decay = 0.1

    # Choice of input features
    in_features_dim = 1

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.98

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 600

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1**(1/100) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch
    batch_num = 8

    # Number of steps per epochs (cannot be None for this dataset)
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each snapshot
    snapshot_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.01
    augment_occlusion = 'none'
    augment_color = 1.0

    # Whether to use loss averaged on all points, or averaged per batch.
    batch_averaged_loss = False

    # Do we nee to save convergence
    saving = True
    saving_path = None


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Initiate the environment
    ##########################

    # If not called with arguments, choose the gpu
    if len(sys.argv) > 2:
        GPU_ID = sys.argv[1]
        log_path = sys.argv[2]
    else:
        # Choose which gpu to use
        GPU_ID = '0'
        log_path = None

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Enable/Disable warnings (set level to '0'/'3')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    config = NPM3DConfig()

    # Modify some of the config parameters for each GPU if called with arguments
    if len(sys.argv) > 2:

        # Mapping from gpu to parameter:
        GPU_to_param = {'0': 64,
                        '1': 64,
                        '2': None,
                        '3': None}

        if GPU_to_param[GPU_ID] is None:
            raise ValueError('GPU_ID not valid')

        # Change param
        config.first_features_dim = GPU_to_param[GPU_ID]
        config.saving_path = log_path
        config.__init__()

        if not config.saving:
            raise ValueError('saving == False in a call from batch')

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    dataset = NPM3DDataset(config.input_threads, load_test=False)

    # Create subsampled input clouds
    dl0 = config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)

    # Initialize input pipelines
    dataset.init_input_pipeline(config)

    # Test the input pipeline alone with this debug function
    # dataset.check_input_pipeline_timing(config)

    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    # Model class
    model = KernelPointFCNN(dataset.flat_inputs, config)

    # Trainer class
    trainer = ModelTrainer(model)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ################
    # Start training
    ################

    print('Start Training')
    print('**************\n')

    trainer.train(model, dataset)






















