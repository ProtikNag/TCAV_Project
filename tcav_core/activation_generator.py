import os
import numpy as np
from .utils import load_image, make_dir_if_not_exists


class ImageActivationGenerator:
    def __init__(self, model, source_dir, activation_dir, target_size=(224, 224), max_examples=100):
        """Initializes the ImageActivationGenerator for generating layer activations.

        Args:
            model (object): A model object with a method to get predictions/activations.
            source_dir (str): Directory containing concept images.
            activation_dir (str): Directory where activations will be stored.
            target_size (tuple): The target size for image resizing (width, height).
            max_examples (int): Maximum number of images to process per concept.
        """
        self.model = model
        self.source_dir = source_dir
        self.activation_dir = activation_dir
        self.target_size = target_size
        self.max_examples = max_examples

        make_dir_if_not_exists(self.activation_dir)

    def generate_activations(self, bottleneck):
        """Generates activations for each concept in source_dir and stores them.

        Args:
            bottleneck (str): The name of the bottleneck layer to extract activations from.
        """
        concepts = os.listdir(self.source_dir)
        print(f"Generating activations for concepts: {concepts}")

        for concept in concepts:
            concept_dir = os.path.join(self.source_dir, concept)
            if not os.path.isdir(concept_dir):
                print(f"Skipping non-directory: {concept_dir}")
                continue

            activations = []
            images = os.listdir(concept_dir)[:self.max_examples]
            print(f"Processing {len(images)} images for concept '{concept}'")

            for image_file in images:
                image_path = os.path.join(concept_dir, image_file)
                image = load_image(image_path, self.target_size)
                activation = self._get_activation(image, bottleneck)
                if activation is not None:
                    activations.append(activation)

            self._save_activations(activations, concept, bottleneck)

    def _get_activation(self, image, bottleneck):
        """Obtains the activation from the specified bottleneck layer for a given image.

        Args:
            image (np.array): Preprocessed image input.
            bottleneck (str): Name of the layer to retrieve activations from.

        Returns:
            np.array: The activation output for the specified layer.
        """
        # Replace 'bottleneck:0' with the exact name of the bottleneck layer's output tensor
        bottleneck_tensor = self.model.sess.graph.get_tensor_by_name(f"{bottleneck}:0")
        activation = self.model.sess.run(bottleneck_tensor,
                                         feed_dict={self.model.sess.graph.get_tensor_by_name("input:0"): image})
        return activation

    def _save_activations(self, activations, concept, bottleneck):
        """Saves the activations to the activation directory as a .npy file.

        Args:
            activations (list): List of activations to be saved.
            concept (str): Concept name for organizing activations.
            bottleneck (str): Name of the bottleneck layer for file naming.
        """
        activations = np.array(activations)
        save_path = os.path.join(self.activation_dir, f"{concept}_{bottleneck}_activations.npy")
        np.save(save_path, activations)
        print(f"Saved activations for concept '{concept}' at layer '{bottleneck}' to: {save_path}")
