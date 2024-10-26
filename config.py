import os

# Directories
WORKING_DIR = './TCAV_Project'
DATA_DIR = os.path.join(WORKING_DIR, 'data')
MODEL_DIR = os.path.join(WORKING_DIR, 'models')
ACTIVATION_DIR = os.path.join(WORKING_DIR, 'activations')
CAV_DIR = os.path.join(WORKING_DIR, 'cavs')
PLOTS_DIR = os.path.join(WORKING_DIR, 'plots')

# Paths
GRAPH_PATH = os.path.join(MODEL_DIR, 'inception5h/tensorflow_inception_graph.pb')
LABEL_PATH = os.path.join(MODEL_DIR, 'inception5h/imagenet_comp_graph_label_strings.txt')

# Experiment parameters
TARGET_CLASS = 'zebra'
CONCEPTS = ["dotted", "striped", "zigzagged"]
BOTTLENECKS = ['mixed4c']
ALPHAS = [0.1]
NUM_RANDOM_EXP = 10
