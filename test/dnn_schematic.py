import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

def draw_dnn(layer_sizes, neuron_colors=None, connection_colors=None, layer_labels=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Set default colors if not provided
    if neuron_colors is None:
        neuron_colors = ['#b2fab4']  # input
        if len(layer_sizes) > 2:
            neuron_colors += ['#dcdcdc'] * (len(layer_sizes) - 2)  # hidden
        neuron_colors += ['#ffcccb']  # output

    if connection_colors is None:
        connection_colors = ['green', 'blue', 'red']

    if layer_labels is None:
        layer_labels = []
        for i in range(len(layer_sizes)):
            if i == 0:
                layer_labels.append("Input Layer \n dropout=0.1 \n Act:ReLu")
            elif i == len(layer_sizes) - 1:
                layer_labels.append("Output Layer, \n Act:sigmoid")
            else:
                layer_labels.append(f"Hidden Layer {i}, \n Dense=32 \n dropout=0.3, \n Act:ReLU")

    v_spacing = 1.5
    h_spacing = 2.5
    max_neurons = max(layer_sizes)

    for i, layer_size in enumerate(layer_sizes):
        x = i * h_spacing
        y_offset = (max_neurons - layer_size) * v_spacing / 2

        visible_neurons = 5  # or 5, depending on space
        for j in range(layer_size):
            y = j * v_spacing + y_offset
            circle = Circle((x, y), radius=0.3, edgecolor="k", facecolor=neuron_colors[i], zorder=4)
            ax.add_patch(circle)
            # Draw connections
            if i > 0:
                color = connection_colors[(i - 1) % len(connection_colors)]
                prev_size = layer_sizes[i - 1]
                prev_offset = (max_neurons - prev_size) * v_spacing / 2
                for k in range(prev_size):
                    prev_y = k * v_spacing + prev_offset
                    ax.plot([x - h_spacing, x], [prev_y, y], color=color, linewidth=0.5, zorder=1)

        # Add text label centered under the layer
        ax.text(x, -1.2, layer_labels[i], ha="center", va="top", fontsize=16)

    ax.set_xlim(-1, h_spacing * len(layer_sizes))
    ax.set_ylim(-2, v_spacing * max_neurons)
    plt.title("DNN Schematic", fontsize=22, loc='center')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),'files','dnn_schematic.pdf'))

# Draw DNN schematic
draw_dnn([14, 32, 32, 1])
