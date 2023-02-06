import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys

import random

# sys.path.extend(['../'])
from . import augment as tools
from . import skcutmix
from . import rotation_aug
from . import skcutmix_pis_new


class Feeder(Dataset):
    def __init__(self, data_path, label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        assert self.data.shape[0]==len(self.label)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)




class Feeder_SameLabelPair(Dataset):
    def __init__(self, 
                data_path, label_path,
                data_path_2, label_path_2,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug

        self.data_path = data_path
        self.label_path = label_path

        self.data_path_2 = data_path_2
        self.label_path_2 = label_path_2

        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap

        self.load_data()
        self.load_data_2()

        self.data2_dt = self.get_class_map(self.data_2, self.label_2)

        print(self.data2_dt)

        # if normalization:
        #     self.get_mean_map()

    @staticmethod
    def get_class_map(data_2, label_2):
        label_list = np.array(label_2)
        class_list = np.unique(label_list)
        dt = {c:[] for c in class_list}
        for idx, c in enumerate(label_list):
            dt[c].append(idx)
        return dt 


    def load_data(self):
        # data: N C V T M
        # return self.data, self.label

        with open(self.label_path, "rb") as f:
            self.sample_name, self.label = pickle.load(f)
        
        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        assert self.data.shape[0]==len(self.label)
        
    def load_data_2(self):
        # data: N C V T M
        # return self.data, self.label

        with open(self.label_path_2, "rb") as f:
            self.sample_name_2, self.label_2 = pickle.load(f)
        
        # load data
        if self.use_mmap:
            self.data_2 = np.load(self.data_path_2, mmap_mode='r')
        else:
            self.data_2 = np.load(self.data_path_2)

        assert self.data_2.shape[0]==len(self.label_2)
    

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self


    def __getitem__(self, index):
        data_numpy_1, label_1 = self.get_data(self.data, self.label, index)

        index2 = np.random.choice(self.data2_dt[label_1])
        data_numpy_2, label_2 = self.get_data(self.data_2, self.label_2, index2)

        assert label_1 == label_2
        return data_numpy_1, data_numpy_2, label_1, index


    def get_data(self, data, label, index):
        data_numpy = data[index]
        label = label[index]
        data_numpy = np.array(data_numpy)

        
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)





class Feeder_SameLabelPair_skcutmix_basic(Feeder_SameLabelPair):

    def __init__(self, 
                data_path, label_path,
                data_path_2, label_path_2, 
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 p1 = 0.5,  p2=0.5, method='',choose_from='src_x_dst',theta=30):
        super(Feeder_SameLabelPair_skcutmix_basic,self).__init__(
            data_path, label_path,
            data_path_2, label_path_2,
            random_choose, random_shift, random_move, window_size, normalization, debug, use_mmap
        )
        self.p1 = p1
        self.p2 = p2 
        self.method=method
        self.choose_from = choose_from
        assert self.choose_from in ['src_x_dst', 'dst_x_dst', 'srcAnddst']
        self.theta = theta

    def __getitem__(self, index):
        data_numpy_1, label_1 = self.get_data(self.data, self.label, index)

        if self.choose_from == 'src_x_dst':
            index2 = np.random.choice(self.data2_dt[label_1])
            data_numpy_2, label_2 = self.get_data(self.data_2, self.label_2, index2)
            assert label_1 == label_2
            data_numpy_aug = self.get_skcutmix_data(self.data[index], self.data_2[index2], self.p1)
            data_numpy_aug_2 = self.get_skcutmix_data(self.data[index], self.data_2[index2], self.p2)
        else:
            raise ValueError()

        th=self.theta
        if th !=0:
            data_numpy_1 = rotation_aug.my_random_rot(data_numpy_1, th)
            data_numpy_2 = rotation_aug.my_random_rot(data_numpy_2, th)
            data_numpy_aug = rotation_aug.my_random_rot(data_numpy_aug, th)
            data_numpy_aug_2 = rotation_aug.my_random_rot(data_numpy_aug_2, th)
        
        return data_numpy_1, data_numpy_2, data_numpy_aug, data_numpy_aug_2, label_1, index
    
    def get_skcutmix_data(self, data1, data2, p):

        data1 = np.array(data1)
        data2 = np.array(data2)

        if self.method=='skcutmix':
            data_numpy = skcutmix.skeleton_cutmix_multiPeople(data1, data2, p)

        elif self.method=='mixup_sameClass':
            alpha = 0.2
            lamda = random.betavariate(alpha, alpha)
            data_numpy = lamda * data1 + (1-lamda) * data2
        
        else:
            raise ValueError()

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy





class Feeder_SameLabelPair_skcutmix_basic_pis_new(Feeder_SameLabelPair):

    def __init__(self, 
                data_path, label_path,
                data_path_2, label_path_2, 
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 p1 = 0.5,  p2=0.5, method='',choose_from='src_x_dst',theta=30,
                 dataset = None, T=1.0
                 ):
        super(Feeder_SameLabelPair_skcutmix_basic_pis_new,self).__init__(
            data_path, label_path,
            data_path_2, label_path_2,
            random_choose, random_shift, random_move, window_size, normalization, debug, use_mmap
        )
        self.p1 = p1
        self.p2 = p2 
        self.method=method
        self.choose_from = choose_from
        assert self.choose_from in ['src_x_dst', 'dst_x_dst', 'srcAnddst']
        self.theta = theta
        self.dataset = dataset
        self.T = T 

    def __getitem__(self, index):
        data_numpy_1, label_1 = self.get_data(self.data, self.label, index)

        if self.choose_from == 'src_x_dst':
            index2 = np.random.choice(self.data2_dt[label_1])
            data_numpy_2, label_2 = self.get_data(self.data_2, self.label_2, index2)
            assert label_1 == label_2
            data_numpy_aug = self.get_skcutmix_data(self.data[index], self.data_2[index2], self.p1, label_1)
            data_numpy_aug_2 = self.get_skcutmix_data(self.data[index], self.data_2[index2], self.p2, label_1)

        else:
            raise ValueError()

        th=self.theta
        if th !=0:
            data_numpy_1 = rotation_aug.my_random_rot(data_numpy_1, th)
            data_numpy_2 = rotation_aug.my_random_rot(data_numpy_2, th)
            data_numpy_aug = rotation_aug.my_random_rot(data_numpy_aug, th)
            data_numpy_aug_2 = rotation_aug.my_random_rot(data_numpy_aug_2, th)
        
        return data_numpy_1, data_numpy_2, data_numpy_aug, data_numpy_aug_2, label_1, index
    
    def get_skcutmix_data(self, data1, data2, p, label=None):

        data1 = np.array(data1)
        data2 = np.array(data2)

        if self.method=='skcutmix_pis':
            data_numpy = skcutmix_pis_new.skeleton_cutmix_multiPeople_pis(data1, data2, p, label, self.dataset, self.T)
        elif self.method=='mixup_sameClass':
            alpha = 0.2
            lamda = random.betavariate(alpha, alpha)
            data_numpy = lamda * data1 + (1-lamda) * data2
        else:
            raise ValueError()

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy




class Feeder_RandomLabelPair(Feeder_SameLabelPair):

    def __init__(self, 
                data_path, label_path,
                data_path_2, label_path_2, 
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 choose_from='src_x_dst',theta=30):
        super(Feeder_RandomLabelPair,self).__init__(
            data_path, label_path,
            data_path_2, label_path_2,
            random_choose, random_shift, random_move, window_size, normalization, debug, use_mmap
        )
        
        self.choose_from = choose_from
        assert self.choose_from in ['src_x_dst', 'dst_x_dst']
        self.theta = theta

    def __getitem__(self, index):
        data_numpy_1, label_1 = self.get_data(self.data, self.label, index)

        if self.choose_from == 'src_x_dst':
            label_2_random = np.random.choice(list(self.data2_dt.keys()))
            index2 = np.random.choice(self.data2_dt[label_2_random])
            data_numpy_2, label_2 = self.get_data(self.data_2, self.label_2, index2)
            assert label_2_random == label_2        
        else:
            raise ValueError()

        th=self.theta
        if th !=0:
            data_numpy_1 = rotation_aug.my_random_rot(data_numpy_1, th)
            data_numpy_2 = rotation_aug.my_random_rot(data_numpy_2, th)
        
        return data_numpy_1, data_numpy_2, label_1, label_2, index
    



class Feeder_PosNegRatioPair(Feeder_SameLabelPair):

    def __init__(self, 
                data_path, label_path,
                data_path_2, label_path_2, 
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 choose_from='src_x_dst',theta=30):
        super(Feeder_PosNegRatioPair,self).__init__(
            data_path, label_path,
            data_path_2, label_path_2,
            random_choose, random_shift, random_move, window_size, normalization, debug, use_mmap
        )
        
        self.choose_from = choose_from
        assert self.choose_from in ['src_x_dst', 'dst_x_dst']
        self.theta = theta


        self.pair_list = self.make_pos_neg_ratio_pair()

    def make_pos_neg_ratio_pair(self):
        '''
        pos/neg
        return pair_list, pair_list[index] = index2
        '''
        pair_list=[]
        n_src = len(self.label)
        n_tgt = len(self.label_2)
        posneg_tick_list=np.random.permutation(np.array(list(range(n_src))) )
        pos_idx_list = posneg_tick_list[:n_src//4]
        neg_idx_list = posneg_tick_list[n_src//4:]

        # pair_list = [index2]
        for i in range(n_src):
            if i in pos_idx_list:
                index = i 
                label_src = self.label[index]
                index2 = np.random.choice(self.data2_dt[label_src])
                pair_list.append( index2 )
            elif i in neg_idx_list:
                index = i 
                label_src = self.label[index]
                label_select_list = list(self.data2_dt.keys())
                label_select_list = [kk for kk in label_select_list if kk!= label_src]
                label_dst = np.random.choice(label_select_list)
                index2 = np.random.choice(self.data2_dt[label_dst])
                pair_list.append( index2 )
            else:
                raise ValueError()

        return pair_list

        



    def __getitem__(self, index):
        data_numpy_1, label_1 = self.get_data(self.data, self.label, index)

        if self.choose_from == 'src_x_dst':
            index2 = self.pair_list[index]
            data_numpy_2, label_2 = self.get_data(self.data_2, self.label_2, index2)     
        else:
            raise ValueError()

        th=self.theta
        if th !=0:
            data_numpy_1 = rotation_aug.my_random_rot(data_numpy_1, th)
            data_numpy_2 = rotation_aug.my_random_rot(data_numpy_2, th)
        
        return data_numpy_1, data_numpy_2, label_1, label_2, index
    


