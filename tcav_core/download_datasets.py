import os
import subprocess
from argparse import ArgumentParser

# Define default values for the dataset
DEFAULT_SOURCE_DIR = '../data/image_net_data'
DEFAULT_NUM_IMAGES = 50
DEFAULT_NUM_RANDOM_FOLDERS = 10
TCAV_REPO_URL = "https://github.com/tensorflow/tcav.git"
TCAV_REPO_DIR = "../data/tcav"


def clone_tcav_repo():
    """Clones the TCAV repository to access dataset scripts."""
    if not os.path.exists(TCAV_REPO_DIR):
        print("Cloning the TCAV repository...")
        subprocess.run(["git", "clone", TCAV_REPO_URL, TCAV_REPO_DIR], check=True)
    else:
        print("TCAV repository already cloned.")


def download_and_prepare_data(source_dir, num_images, num_random_folders):
    """Downloads and prepares datasets for TCAV experiments."""
    clone_tcav_repo()

    # Run the dataset preparation script in the TCAV repository
    script_path = os.path.join(TCAV_REPO_DIR, "tcav", "tcav_examples", "image_models", "imagenet")
    os.chdir(script_path)

    download_cmd = [
        "python", "download_and_make_datasets.py",
        f"--source_dir={source_dir}",
        f"--number_of_images_per_folder={num_images}",
        f"--number_of_random_folders={num_random_folders}"
    ]
    print("Downloading and preparing data...")
    subprocess.run(download_cmd, check=True)

    # Copy relevant model files to accessible paths for main project
    model_files = ["mobilenet_v2_1.0_224", "inception5h"]
    for model_file in model_files:
        src_path = os.path.join(source_dir, model_file)
        dest_path = os.path.join(TCAV_REPO_DIR, model_file)
        if os.path.exists(src_path):
            subprocess.run(["cp", "-av", src_path, dest_path], check=True)
            subprocess.run(["rm", "-r", src_path], check=True)
            print(f"Moved {model_file} to project root.")
        else:
            print(f"Warning: Model file {model_file} not found in source directory.")

    print("Data download and preparation complete.")


if __name__ == '__main__':
    parser = ArgumentParser(description="Download and prepare datasets for TCAV experiments.")
    parser.add_argument("--source_dir", type=str, default=DEFAULT_SOURCE_DIR, help="Directory to save downloaded data")
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES, help="Number of images per folder")
    parser.add_argument("--num_random_folders", type=int, default=DEFAULT_NUM_RANDOM_FOLDERS,
                        help="Number of random folders")

    args = parser.parse_args()
    download_and_prepare_data(args.source_dir, args.num_images, args.num_random_folders)
