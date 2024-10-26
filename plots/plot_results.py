import matplotlib.pyplot as plt


def plot_results(results, num_random_exp=10):
    """Plots TCAV results.

    Args:
        results (list of dict): List of results from TCAV analysis. Each entry should contain:
            - 'bottleneck': The name of the bottleneck layer.
            - 'concept': The concept name.
            - 'tcav_score': The TCAV score (float).
        num_random_exp (int): The number of random experiments conducted.
    """
    # Organize results by bottleneck layer
    bottleneck_results = {}
    for result in results:
        bottleneck = result['bottleneck']
        concept = result['concept']
        tcav_score = result['tcav_score']

        if bottleneck not in bottleneck_results:
            bottleneck_results[bottleneck] = []
        bottleneck_results[bottleneck].append((concept, tcav_score))

    # Plot results for each bottleneck layer
    for bottleneck, concept_scores in bottleneck_results.items():
        concepts, scores = zip(*concept_scores)

        plt.figure(figsize=(10, 6))
        plt.barh(concepts, scores, color='skyblue')
        plt.xlabel('TCAV Score')
        plt.title(f"TCAV Scores for Bottleneck Layer: {bottleneck}\n(Target Class: Influence of Concepts)")
        plt.xlim(0, 1)

        # Annotate bars with TCAV scores
        for index, score in enumerate(scores):
            plt.text(score + 0.02, index, f"{score:.2f}", va='center')

        plt.tight_layout()
        plt.show()
