import numpy as np
from representation.representation import MlpRepresentation


def representation_matrix(x_data, model, device):
    r = MlpRepresentation(model=model, device=device)
    _, weights, r_y = r.forward(x_data)
    return weights, r_y


def matrix_l2_norm(matrix):
    m = matrix.numpy()
    flatten_matrix = m.flatten()
    return np.linalg.norm(flatten_matrix)


def norm_comparison(norm1, norm2):
    return np.abs(norm1-norm2)


def matrix_comparison(matrix_1, matrix_2):
    norm1 = matrix_l2_norm(matrix_1)
    norm2 = matrix_l2_norm(matrix_2)
    return norm_comparison(norm1, norm2)


def layers_output(x, weights):
    outputs_of_layers = []
    vector_ones = np.ones(len(x))
    a = weights[0].numpy().dot(vector_ones)
    outputs_of_layers.append(a)
    for i in range(1, len(weights)):
        a = weights[i].numpy().dot(a)
        outputs_of_layers.append(a)
    return outputs_of_layers


def norm(vector):
    norm_vector = np.linalg.norm(vector)
    return norm_vector

