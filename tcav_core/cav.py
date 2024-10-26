import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .utils import save_cavs


class CAV:
    def __init__(self, concept, activations, alphas):
        """Initializes a CAV for a concept.

        Args:
            concept (str): The concept name.
            activations (dict): A dictionary with two keys:
                - 'concept': A list of activations for the concept.
                - 'random': A list of activations for random counterexamples.
            alphas (list): List of regularization parameters for the classifier.
        """
        self.concept = concept
        self.activations = activations
        self.alphas = alphas
        self.vector = None

    def train_cav(self):
        """Trains the CAV by fitting a linear classifier between concept and random activations."""
        # Combine activations and prepare labels
        concept_activations = self.activations['concept']
        random_activations = self.activations['random']

        X = np.vstack([concept_activations, random_activations])
        y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

        # Standardize the activations
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train logistic regression model with each alpha and select the best-performing CAV
        best_score = -np.inf
        best_vector = None

        for alpha in self.alphas:
            clf = LogisticRegression(C=1.0 / alpha, solver='liblinear', max_iter=1000)
            clf.fit(X, y)

            # Take the normal vector to the decision boundary as the CAV direction
            score = clf.score(X, y)
            if score > best_score:
                best_score = score
                best_vector = clf.coef_.flatten()  # CAV vector

        self.vector = best_vector
        print(f"CAV for concept '{self.concept}' trained with score: {best_score}")
        return self

    def save(self, save_dir):
        """Saves the trained CAV to a specified directory.

        Args:
            save_dir (str): Directory where the CAV data will be saved.
        """
        cav_data = {
            'concept': self.concept,
            'vector': self.vector,
            'alphas': self.alphas
        }
        save_cavs(cav_data, save_dir, self.concept)
