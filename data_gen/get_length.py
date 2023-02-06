import os,sys
import numpy as np 
import pickle
from IPython import embed 

import shutil

def load_pkl(fname):
    return pickle.load(open(fname,"rb"))

def write_pkl(fname, data):
    with open(fname,"wb") as f:
        pickle.dump(data, f)


def combine_etri_adult_elder():


    src_dir_list=["/home/yqs/liuhc/action/data/etri220429",
             "/home/yqs/liuhc/action/data/etri220501_resample64"]

    for src_dir in src_dir_list:
        for split in ['train', 'test']:

            combine_files(
                src_dir + f"/adult/{split}_data_joint_len64.npy",
                src_dir + f"/adult/{split}_label.pkl",
                src_dir + f"/elder/{split}_data_joint_len64.npy",
                src_dir + f"/elder/{split}_label.pkl",
                src_dir + f"/adult_elder/{split}_data_joint_len64.npy",
                src_dir + f"/adult_elder/{split}_label.pkl",
            )
            # combine_files(
            #     src_dir + "/adult/train_data_joint_len64.npy",
            #     src_dir + "/adult/train_label.pkl",
            #     src_dir + "/elder/train_data_joint_len64.npy",
            #     src_dir + "/elder/train_label.pkl",
            #     src_dir + "/adult_elder/train_data_joint_len64.npy",
            #     src_dir + "/adult_elder/train_label.pkl",
            # )

    

def combine_files(data_path_1, label_path_1, data_path_2, label_path_2,
    output_data_path, output_label_path):

    data_1 = np.load(data_path_1)
    data_2 = np.load(data_path_2)
    data   = np.concatenate([data_1, data_2],0)

    a1,b1=load_pkl(label_path_1)
    a2,b2=load_pkl(label_path_2)
    res = [a1+a2, b1+b2]

    assert len(data)==len(res[0])==len(res[1])
    
    np.save(output_data_path, data)
    write_pkl(output_label_path, res)



def get_ntu_shape():

    import glob
    from tqdm import tqdm
    ntu_raw_file_path = "/home/yqs/liuhc/action_data/ntu/nturgb+d_skeletons"
    file_list = glob.glob(os.path.join(ntu_raw_file_path, "*.skeleton"))
    print(len(file_list))

    def read_skeleton(file):
        with open(file, 'r') as f:
            numFrame = int(f.readline())
        return numFrame
    
    num_frame_dt = {}
    for path in tqdm(file_list):
        num_frame = read_skeleton(path)
        name = os.path.basename(path).replace(".skeleton", "")
        num_frame_dt[name] = num_frame

    print(len(num_frame_dt))
    save_path="/home/yqs/liuhc/action/data/ntu220414/stats/frame_num.pkl"
    write_pkl(save_path, num_frame_dt)
        

def get_etri_shape():
    
    import glob
    from tqdm import tqdm
    import pandas as pd 
    etri_raw_file_path = "/home/yqs/liuhc/action_data/etri_data"
    file_list = glob.glob(os.path.join(etri_raw_file_path, "P001-P050/*.csv")) + \
                glob.glob(os.path.join(etri_raw_file_path, "P051-P100/*.csv"))
    print(len(file_list))


    def etri_read_skeleton(file):
        x=pd.read_csv(file)

        if 'frameNum' not in x.columns:
            print(file, 'not valid frameNum')
            return 0

        x = x.dropna(subset=['bodyindexID'],how='any')
        # print(x)

        body_index_list = np.unique(x['bodyindexID'].tolist())
        # frame_index_list = np.sort(np.unique(x['frameNum'].tolist()))
        frame_index_list = np.unique(x['frameNum'].tolist())
        return len(frame_index_list)

    num_frame_dt = {}
    for path in tqdm(file_list):
        num_frame = etri_read_skeleton(path)
        name = os.path.basename(path).replace(".csv", "")
        num_frame_dt[name] = num_frame


    print(len(num_frame_dt))
    save_path="/home/yqs/liuhc/action/data/etri220429/stats/frame_num.pkl"
    write_pkl(save_path, num_frame_dt)

    


def main():

    get_ntu_shape()
    get_etri_shape()
    
    # combine adult and elder split for ntu<->etri cross-dataset setting.
    # combine_etri_adult_elder()

    



if __name__ == "__main__":
    main()