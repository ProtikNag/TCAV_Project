import os
import numpy as np
from .utils import make_dir_if_not_exists, save_cavs, load_cavs
from .cav import CAV  # This assumes you'll have a CAV class defined in cav.py


class TCAVAnalysis:
    def __init__(self, model, target, concepts, bottlenecks, act_generator, alphas, cav_dir, num_random_exp=10):
        """Initializes the TCAV analysis for a target class with specified concepts.

        Args:
            model (object): The model wrapper with prediction functions.
            target (str): The target class to analyze.
            concepts (list): List of concept names to evaluate.
            bottlenecks (list): List of layers (bottlenecks) for which to compute activations.
            act_generator (object): Activation generator for obtaining activations.
            alphas (list): List of regularization strengths for CAVs.
            cav_dir (str): Directory to save and load CAV files.
            num_random_exp (int): Number of random experiments for statistical significance.
        """
        self.model = model
        self.target = target
        self.concepts = concepts
        self.bottlenecks = bottlenecks
        self.act_generator = act_generator
        self.alphas = alphas
        self.cav_dir = cav_dir
        self.num_random_exp = num_random_exp

        make_dir_if_not_exists(self.cav_dir)

    def run(self, run_parallel=False):
        """Runs the TCAV analysis.

        Args:
            run_parallel (bool): Whether to run in parallel mode (if supported).

        Returns:
            List[Dict]: A list of results with TCAV scores for each concept and bottleneck.
        """
        results = []

        for bottleneck in self.bottlenecks:
            print(f"Processing bottleneck: {bottleneck}")
            for concept in self.concepts:
                # Generate and load CAV
                cav = self._get_or_train_cav(concept, bottleneck)

                # Calculate TCAV score
                tcav_score = self._calculate_tcav_score(bottleneck, concept, cav)

                # Append results
                results.append({
                    'bottleneck': bottleneck,
                    'concept': concept,
                    'tcav_score': tcav_score
                })

        return results

    def _get_or_train_cav(self, concept, bottleneck):
        """Gets or trains a CAV for a given concept and bottleneck.

        Args:
            concept (str): The concept name.
            bottleneck (str): The bottleneck layer name.

        Returns:
            CAV: The trained or loaded CAV.
        """
        cav_path = os.path.join(self.cav_dir, f"{concept}_{bottleneck}_cav.pkl")

        if os.path.exists(cav_path):
            print(f"Loading CAV from: {cav_path}")
            return load_cavs(cav_path)

        print(f"Training CAV for concept '{concept}' at bottleneck '{bottleneck}'")
        activations = self.act_generator.generate_activations(bottleneck)

        # Train CAV (CAV class should handle this)
        cav = CAV(concept, activations, self.alphas)
        save_cavs(cav, self.cav_dir, concept)

        return cav

    def _calculate_tcav_score(self, bottleneck, concept, cav):
        """Calculates the TCAV score for a concept by evaluating model sensitivity.

        Args:
            bottleneck (str): The bottleneck layer name.
            concept (str): The concept name.
            cav (CAV): The CAV object.

        Returns:
            float: The TCAV score.
        """
        sensitivity_count = 0
        total_samples = 0

        # Run sensitivity test for each example in the target class
        target_activations = self.act_generator.generate_activations(bottleneck)

        for activation in target_activations:
            sensitivity = np.dot(cav.vector, activation)
            if sensitivity > 0:
                sensitivity_count += 1
            total_samples += 1

        tcav_score = sensitivity_count / total_samples if total_samples > 0 else 0
        print(f"TCAV score for concept '{concept}' at bottleneck '{bottleneck}': {tcav_score}")
        return tcav_score
