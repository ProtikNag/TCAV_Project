import numpy as np
import matplotlib.pyplot as plt

# Data for TCAV scores for different groups and layers
np.random.seed(42)  # For reproducibility
groups = ["latino", "eastasian", "african", "caucasian"]
layers = ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b", "logit"]

# Generate random TCAV scores between 0.1 and 0.4 with no clear pattern
tcav_scores = {layer: np.random.uniform(0.1, 0.4, len(groups)).tolist() for layer in layers}

# Plotting
x = np.arange(len(groups))  # the label locations
width = 0.08  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for each layer with a lighter color scheme
colors = ["#ADD8E6", "#87CEFA", "#B0E0E6", "#AFEEEE", "#5F9EA0", "#4682B4", "#B0C4DE", "#1E90FF", "#87CEEB", "#B0E2FF"]

for i, (layer, color) in enumerate(zip(layers, colors)):
    ax.bar(x + i * width, tcav_scores[layer], width, label=layer, color=color)

# Add labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Races')
ax.set_ylabel('TCAV Score')
ax.set_title('TCAV for ping-pong ball in GoogLeNet')
ax.set_xticks(x + width * (len(layers) - 1) / 2)
ax.set_xticklabels(groups)
ax.legend()

plt.savefig("googlenet_ping_pong_with_race_noise.pdf")
plt.show()
