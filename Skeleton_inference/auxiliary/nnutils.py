import numpy as np
import cv2
import random
import math
import os
import sys
import scipy.io as sio
import torch

def define_lines(num_points=600, nb_primitives=20):
    grain = int(num_points / nb_primitives) - 1
    grain = grain * 1.0
    print(grain)

    vertices = []
    for i in range(0, int(grain + 1)):
        vertices.append([i / grain, 0])
    grid_list = [vertices for i in range(0, nb_primitives)]

    lines = []
    for i in range(0, int(grain)):
        lines.append([i, i + 1])
    lines_array = np.array(lines)
    lines_array = lines_array.astype(int)

    lines_adjacent = []
    lines_adjacent.append([1, 1])
    for i in range(1, int(grain)):
        lines_adjacent.append([i - 1, i + 1])
    lines_adjacent.append([int(grain - 1), int(grain - 1)])
    lines_adjacent_tensor = torch.cuda.LongTensor(lines_adjacent)
    return grid_list, lines_array, lines_adjacent_tensor

def define_squares(num_points=2000, nb_primitives=20):
    grain = int(np.sqrt(num_points / nb_primitives)) - 1
    grain = grain * 1.0
    print(grain)

    vertices = []
    for i in range(0, int(grain + 1)):
        for j in range(0, int(grain + 1)):
            vertices.append([i / grain, j / grain])
    grid_list = [vertices for i in range(0, nb_primitives)]

    faces = []
    prim = 0
    for i in range(1, int(grain + 1)):
        for j in range(0, int(grain + 1) - 1):
            faces.append([(grain + 1) * (grain + 1) * prim + j + (grain + 1) * i,
                          (grain + 1) * (grain + 1) * prim + j + (grain + 1) * i + 1,
                          (grain + 1) * (grain + 1) * prim + j + (grain + 1) * (i - 1)])
    for i in range(0, int(grain + 1) - 1):
        for j in range(1, int(grain + 1)):
            faces.append([(grain + 1) * (grain + 1) * prim + j + (grain + 1) * i,
                          (grain + 1) * (grain + 1) * prim + j + (grain + 1) * i - 1,
                          (grain + 1) * (grain + 1) * prim + j + (grain + 1) * (i + 1)])
    faces_array = np.array(faces)
    faces_array = faces_array.astype(int)

    edge = []
    for i, j in enumerate(faces_array):
        edge.append(j[:2])
        edge.append(j[1:])
        edge.append(j[[0, 2]])
    edge = np.array(edge)
    edge_im = edge[:, 0] * edge[:, 1] + (edge[:, 0] + edge[:, 1]) * 1j
    unique = np.unique(edge_im, return_index=True)[1]
    edge_unique = edge[unique]

    vertex_adj_matrix=[]
    for i in range(0, len(vertices)):
        vertex_adj=edge_unique[np.where((edge_unique==i).astype(int).sum(axis=1)==1)[0]].reshape(-1)
        i_index=np.where(vertex_adj!=i)[0]
        vertex_adj=vertex_adj[i_index]
        vertex_adj=vertex_adj.repeat(12/len(vertex_adj))
        vertex_adj_matrix.append(vertex_adj)
    vertex_adj_matrix = np.array(vertex_adj_matrix)
    vertex_adj_matrix_tensor = torch.from_numpy(vertex_adj_matrix).type(torch.cuda.LongTensor)
    return grid_list, faces_array, vertex_adj_matrix_tensor

def curve_laplacian(pointsReconstructed, nb_primitives, lines_adjacent_tensor):
    pointsReconstructed_line = pointsReconstructed.view(pointsReconstructed.size()[0] * nb_primitives, \
        pointsReconstructed.size()[1] // nb_primitives, pointsReconstructed.size()[2])
    vertex_adjacent_coor = torch.index_select(pointsReconstructed_line, 1, lines_adjacent_tensor.view(-1)). \
        view(pointsReconstructed_line.size()[0], lines_adjacent_tensor.size()[0], 
        lines_adjacent_tensor.size()[1], pointsReconstructed_line.size()[2])
    vertex_adjacent_coor_mean = torch.mean(vertex_adjacent_coor, 2).squeeze(2)
    laplacian_smooth = torch.abs(pointsReconstructed_line - vertex_adjacent_coor_mean)
    laplacian_smooth = laplacian_smooth.view(laplacian_smooth.size()[0] // nb_primitives,
        laplacian_smooth.size()[1] * nb_primitives, laplacian_smooth.size()[2])
    return laplacian_smooth

def surface_laplacian(pointsReconstructed, nb_primitives, vertex_adj_matrix_tensor):
    pointsReconstructed_square = pointsReconstructed.view(pointsReconstructed.size()[0] * nb_primitives,
        pointsReconstructed.size()[1] // nb_primitives, pointsReconstructed.size()[2])
    vertex_adj_coor = torch.index_select(pointsReconstructed_square, 1, vertex_adj_matrix_tensor.view(-1)).view(
        pointsReconstructed_square.size()[0], vertex_adj_matrix_tensor.size()[0], vertex_adj_matrix_tensor.size()[1],
        pointsReconstructed_square.size()[2])
    vertex_adj_coor_mean = torch.mean(vertex_adj_coor, 2).squeeze(2)
    laplacian_smooth = torch.abs(pointsReconstructed_square - vertex_adj_coor_mean)
    laplacian_smooth = laplacian_smooth.view(laplacian_smooth.size()[0] // nb_primitives,
        laplacian_smooth.size()[1] * nb_primitives, laplacian_smooth.size()[2])
    return laplacian_smooth