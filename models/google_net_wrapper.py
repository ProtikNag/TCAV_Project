import tensorflow as tf
from tcav_core.utils import create_session


class GoogleNetWrapper:
    def __init__(self, graph_path, label_path):
        """Initializes the GoogleNet model with a given graph and labels.

        Args:
            graph_path (str): Path to the frozen model graph (.pb file).
            label_path (str): Path to the label file containing class names.
        """
        self.graph_path = graph_path
        self.label_path = label_path
        self.sess = self._load_model()
        self.labels = self._load_labels()

    def _load_model(self):
        """Loads the TensorFlow model graph and initializes the session."""
        print("Loading GoogleNet model from:", self.graph_path)
        sess = create_session()

        with tf.io.gfile.GFile(self.graph_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        return sess

    def _load_labels(self):
        """Loads the labels from the label file."""
        print("Loading labels from:", self.label_path)
        with open(self.label_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def get_predictions(self, image):
        """Runs inference on a single image and returns the predictions.

        Args:
            image (np.array): Preprocessed image array.

        Returns:
            List[Tuple[str, float]]: List of tuples containing class label and probability.
        """
        # Assuming the input and output tensor names are known.
        input_tensor = self.sess.graph.get_tensor_by_name("input:0")
        output_tensor = self.sess.graph.get_tensor_by_name("softmax:0")

        predictions = self.sess.run(output_tensor, feed_dict={input_tensor: image})
        predictions = predictions[0]  # Remove batch dimension if exists

        # Pair each label with its corresponding prediction probability
        return [(self.labels[i], predictions[i]) for i in range(len(predictions))]

    def close(self):
        """Closes the TensorFlow session."""
        self.sess.close()
