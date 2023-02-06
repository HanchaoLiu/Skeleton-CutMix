import argparse
import pickle
from tqdm import tqdm
import sys

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

from IPython import embed 
import shutil

max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

target_frame=64

import numpy as np
import os

import pandas as pd 

import glob 




def load_pkl(fname):
    return pickle.load(open(fname,"rb"))

def write_pkl(fname, data):
    with open(fname,"wb") as f:
        pickle.dump(data, f)




def ntu_gendata_length_multiprocessing(data_path, out_path, benchmark='xview', part='eval'):
    '''
    save to os.path.join(out_path, out_dir)
    '''
    
    original_base_path= data_path
    original_data_path = os.path.join(original_base_path, benchmark, f'{part}_data_joint.npy')
    original_label_path = os.path.join(original_base_path, benchmark, f'{part}_label.pkl')

    original_data = np.load(original_data_path, mmap_mode='r')
    original_label = pickle.load(open(original_label_path,"rb"))

    sample_name, sample_label = original_label

    assert original_label[0]==sample_name
    assert original_data.shape[2]==300
    assert original_data.shape[0]==len(sample_label)

    # get num dt 
    frame_num_dt_path = "/home/yqs/liuhc/action/data/ntu220414/stats/frame_num.pkl"
    frame_num_dt = load_pkl(frame_num_dt_path)


    # fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    fp = np.zeros((len(sample_label), 3, target_frame, num_joint, max_body_true), dtype=np.float32)
    for i, s in enumerate(tqdm(sample_name)):
        
        if s.endswith('.skeleton'):
            s = s.replace('.skeleton', '')
        n_frames = frame_num_dt[s]
        data = np.array(original_data[i])

        # (c,n_frames,v,m) -> (c,64,v,m)
        n_frames = min(n_frames, 300)
        fp[i,:,:,:,:] = slice_temporal_fn(data, n_frames, target_frame)

    #     t_max = min(data.shape[1], max_frame)
    #     fp[i, :, 0:t_max, :, :] = data[:,:t_max,:,:]

    # fp = pre_normalization(fp)
    np.save('{}/{}_data_joint_len64.npy'.format(out_path, part), fp)
    print(fp.shape)
    print("saved to",'{}/{}_data_joint_len64.npy'.format(out_path, part) )

    save_label_path=os.path.join(out_path, f'{part}_label.pkl')
    shutil.copy(original_label_path, save_label_path)
    print("save to", save_label_path)



def slice_temporal_fn(data_numpy, n_frames, target_frame):
    assert target_frame==64

    C,T,V,M=data_numpy.shape
    # n_frames = self.T
    # n_frames = length of action 
    # c,t,v
    output_n=target_frame
    select_idx=np.linspace(0,n_frames-1,num=output_n)
    select_idx=np.round(select_idx).astype(int)
    data_numpy=data_numpy[:,select_idx,:,:]
    return data_numpy





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/ntu/')
    arg = parser.parse_args()

    # task: from data_path get data and label 
    # get frame num
    # interpolate data to data64
    # save to out_folder.

    arg.data_path = "/home/yqs/liuhc/action/data/ntu220414"
    arg.out_folder = "/home/yqs/liuhc/action/data/ntu220502_resample64"
    if not os.path.exists(arg.out_folder):
        os.mkdir(arg.out_folder)


    # benchmark = ['adult', 'elder', 'all_ages']
    # part = ['train', 'test']

    benchmark = ['xsub']
    part = ['val','train']

    # benchmark = ['debug']
    # part = ['test']
    

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            print("to path = ", out_path)
            # etri_gendata(
            #     arg.data_path,
            #     out_path,
            #     benchmark=b,
            #     part=p)

            ntu_gendata_length_multiprocessing(
                arg.data_path,
                out_path, 
                benchmark=b,
                part=p
            )
