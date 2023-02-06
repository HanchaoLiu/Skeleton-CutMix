import sys

sys.path.extend(['../'])
from graph import tools

# num_node = 25
# self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


# remove (5,21), (9,21), (17,1), (13,1)
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3),  (6, 5), (7, 6),
#                     (8, 7), (10, 9), (11, 10), (12, 11), 
#                     (14, 13), (15, 14), (16, 15),(18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

num_node = 13
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(i,i+1) for i in range(num_node-1)]
inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A







class Graph_full:
    def __init__(self, labeling_mode='spatial'):

        num_node = 25
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                            (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                            (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.num_points = num_node
        self.joint_idx_list = list(range(25))

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A



class Graph_hand:
    def __init__(self, labeling_mode='spatial'):

        num_node = 13
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(i,i+1) for i in range(num_node-1)]
        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.num_points = num_node
        self.joint_idx_list = [21,22,7,6,5,4,20,8,9,10,11,24,23]

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


class Graph_foot:
    def __init__(self, labeling_mode='spatial'):

        num_node = 9
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(i,i+1) for i in range(num_node-1)]
        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

        self.num_points = num_node
        self.joint_idx_list = [15,14,13,12,0,16,17,18,19]

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


class Graph_arm:
    def __init__(self, labeling_mode='spatial'):

        num_node = 7
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(i,i+1) for i in range(num_node-1)]
        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.num_points = num_node
        self.joint_idx_list = [6,5,4,20,8,9,10]

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A




class Graph_handOnly:
    def __init__(self, labeling_mode='spatial'):

        num_node = 8
        self_link = [(i, i) for i in range(num_node)]
        inward_ori_index = [(0,1),(1,2),(2,3),(7,6),(6,5),(5,4)]
        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]
        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.num_points = num_node
        self.joint_idx_list = [21,22,7,6,10,11,24,23]

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A


class Graph_torsoHead:
    def __init__(self, labeling_mode='spatial'):

        num_node = 5
        inward_ori_index = [(i,i+1) for i in range(num_node-1)]
        
        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]

        self_link = [(i, i) for i in range(num_node)]


        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.num_points = num_node
        self.joint_idx_list = [0,1,20,2,3]

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A



class Graph_torso_head:
    def __init__(self, labeling_mode='spatial'):

        num_node = 4
        inward_ori_index = [(i,i+1) for i in range(num_node-1)]
        
        inward = [(i, j) for (i, j) in inward_ori_index]
        outward = [(j, i) for (i, j) in inward]

        self_link = [(i, i) for i in range(num_node)]


        neighbor = inward + outward

        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
