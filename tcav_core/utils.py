import os
import tensorflow as tf


def create_session():
    """Creates a TensorFlow session with specific configurations.

    Returns:
        tf.compat.v1.Session: A TensorFlow session with GPU growth option enabled.
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # Prevent TensorFlow from allocating the entire GPU memory
    sess = tf.compat.v1.Session(config=config)
    return sess


def make_dir_if_not_exists(directory):
    """Creates a directory if it does not exist.

    Args:
        directory (str): Path of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def load_image(image_path, target_size):
    """Loads and preprocesses an image for model input.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image (width, height).

    Returns:
        np.array: The preprocessed image array ready for model input.
    """
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def save_cavs(cav_data, save_dir, concept_name):
    """Saves Concept Activation Vectors (CAVs) to the specified directory.

    Args:
        cav_data (dict): A dictionary containing CAV data.
        save_dir (str): Directory where CAV data should be saved.
        concept_name (str): Name of the concept being saved.
    """
    import pickle

    make_dir_if_not_exists(save_dir)
    cav_file = os.path.join(save_dir, f"{concept_name}_cav.pkl")

    with open(cav_file, "wb") as f:
        pickle.dump(cav_data, f)
    print(f"CAV for concept '{concept_name}' saved at: {cav_file}")


def load_cavs(cav_file):
    """Loads a saved Concept Activation Vector (CAV) file.

    Args:
        cav_file (str): Path to the CAV file.

    Returns:
        dict: Loaded CAV data.
    """
    import pickle

    with open(cav_file, "rb") as f:
        cav_data = pickle.load(f)
    print(f"CAV loaded from: {cav_file}")
    return cav_data
