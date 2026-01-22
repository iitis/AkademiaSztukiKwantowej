import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler



def randomize_points_in_rectangle(corner1, corner2, num_points=100, random_state=None):
    """
    Generate a random set of points within a rectangle defined by two opposing corners.

    Parameters:
        corner1 (tuple): Coordinates (x1, y1) of the first corner.
        corner2 (tuple): Coordinates (x2, y2) of the opposing corner.
        num_points (int): The number of random points to generate.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        numpy.ndarray: Array of shape (num_points, 2) containing random points.
    """
    # Set random seed for reproducibility if specified
    if random_state is not None:
        np.random.seed(random_state)

    # Extract the rectangle's bounds
    x_min, x_max = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
    y_min, y_max = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])

    # Generate random points within the rectangle
    x_coords = np.random.uniform(x_min, x_max, num_points)
    y_coords = np.random.uniform(y_min, y_max, num_points)

    # Combine x and y coordinates
    points = np.column_stack((x_coords, y_coords))
    return points


def showXy(X, y, finish=False, ofname=None):
    if finish:
        plt.figure('data')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

    # Visualize the points
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=25, c='b', edgecolor='w', 
                alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=25, c='r', edgecolor='w', 
                alpha=0.5)
    #plt.title(f"Random Points in Rectangle ({corner1}, {corner2})")
    if finish:
        plt.title('Przykładowe dane')
        plt.xlabel("Cecha pierwsza")
        plt.ylabel("Cecha druga")
        #plt.axis("equal")
        #plt.grid(True)
        #plt.tight_layout()
        if ofname is not None:
            plt.savefig(ofname + '_data.png')
            #plt.close('all')
        else:
            plt.show()


def plot_mlp_heatmap(clf, xlim, ylim, grid_resolution=100, X=None, yc=None,
                     ofname=None):
    """
    Plots a heatmap of neuron outputs for a given sklearn MLP classifier.

    Parameters:
        clf (MLPClassifier): The trained MLP classifier.
        xlim (tuple): The x-axis range as (xmin, xmax).
        ylim (tuple): The y-axis range as (ymin, ymax).
        grid_resolution (int): The resolution of the grid for heatmap.

    Returns:
        None: Displays the heatmap.
    """
    # Create a grid of points within the specified x and y limits
    x = np.linspace(xlim[0], xlim[1], grid_resolution)
    y = np.linspace(ylim[0], ylim[1], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get the activations of neurons in the hidden layers
    # Using `predict_proba` or `decision_function` as a proxy for neuron outputs
    neuron_outputs = clf.predict_proba(grid_points)[:, 1].reshape(xx.shape)

    # Plot heatmap
    plt.figure('output')#figsize=(10, 6))
    plt.contourf(xx, yy, neuron_outputs, levels=100, cmap="gnuplot", alpha=0.6)
    plt.colorbar(label="Odpowiedź neuronu (szacowane prawdopodobieństwo klasy 1)")
    plt.title("Mapa ciepła odpowiedzi neuronu wyjściowego")
    if X is not None:
        showXy(X, yc)
    plt.xlabel("Cecha pierwsza")
    plt.ylabel("Cecha druga")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    #plt.tight_layout()
    if ofname is not None:
        plt.savefig(ofname + '_output.png')
        #plt.close('all')
    else:
        plt.show()


def extract_hidden_layer_outputs(clf, X):
    """
    Extract the activations of the hidden layers for the input data.

    Parameters:
        clf (MLPClassifier): The trained MLP classifier.
        X (numpy.ndarray): Input data for which to calculate activations.

    Returns:
        List[numpy.ndarray]: Activations of each hidden layer.
    """
    # Use the private method `_predict` to extract hidden layer outputs
    activations = []
    hidden_layer_sizes = clf.hidden_layer_sizes
    layer_inputs = X
    for coefs, intercepts in zip(clf.coefs_[:-1], clf.intercepts_[:-1]):
        layer_inputs = layer_inputs @ coefs + intercepts
        #layer_inputs = np.tanh(layer_inputs)  # Activation function (tanh)
        layer_inputs = np.maximum(layer_inputs, 0)  # Activation function (ReLu)
        activations.append(layer_inputs)
    return activations

def plot_hidden_layer_heatmap(clf, xlim, ylim, grid_resolution=100, 
                              X=None, yc=None, ofname=None):
    """
    Plot heatmaps of hidden layer neuron outputs for a given MLPClassifier.

    Parameters:
        clf (MLPClassifier): The trained MLP classifier.
        xlim (tuple): The x-axis range as (xmin, xmax).
        ylim (tuple): The y-axis range as (ymin, ymax).
        grid_resolution (int): The resolution of the grid for heatmap.

    Returns:
        None: Displays the heatmap for each neuron in the hidden layers.
    """
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], grid_resolution)
    y = np.linspace(ylim[0], ylim[1], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get activations for each hidden layer
    hidden_activations = extract_hidden_layer_outputs(clf, grid_points)

    # Plot each neuron in each hidden layer
    for layer_idx, layer_output in enumerate(hidden_activations):
        for neuron_idx in range(layer_output.shape[1]):
            plt.figure()#figsize=(10, 6))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            neuron_output = layer_output[:, neuron_idx].reshape(xx.shape)
            plt.contourf(xx, yy, neuron_output, levels=100, cmap="gnuplot", alpha=0.6)
            plt.colorbar(label=f"Odpowiedź neuronu (warstwa {layer_idx + 1}, neuron {neuron_idx + 1})")
            plt.title(f"Mapa ciepła odpowiedzi neuronu - warstwa {layer_idx + 1}, neuron {neuron_idx + 1}")
            plt.xlabel("Cecha pierwsza")
            plt.ylabel("Cecha druga")
            if X is not None:
                showXy(X, yc)
            #plt.grid(True)
            #plt.tight_layout()
            if ofname is not None:
                plt.savefig(f'{ofname}_l{layer_idx}_n{neuron_idx}.png')
                #plt.close('all')
            else:
                plt.show()


def generate(corners, random_state=None):
    Xs, ys = [], []
    for c1, c2, n, y in corners:
        X = randomize_points_in_rectangle(c1, c2, n, random_state)
        Xs.append(X)
        ys.append(np.ones(len(X)) * y)
    return np.vstack(Xs), np.hstack(ys)
