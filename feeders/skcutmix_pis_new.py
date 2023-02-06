import os,sys
import numpy as np 
from IPython import embed
import pickle



# inward to outward, from root(idx=1) to end effector.
# bone_idx: [inward_joint_idx, outward_joint_idx]
bones_dt={
    0:[1,0],
    1:[20,2],
    2:[2,3],
    3:[20,4],
    4:[4,5],
    5:[5,6],
    6:[6,7],
    7:[20,8],
    8:[8,9],
    9:[9,10],
    10:[10,11],
    11:[0,12],
    12:[12,13],
    13:[13,14],
    14:[14,15],
    15:[0,16],
    16:[16,17],
    17:[17,18],
    18:[18,19],
    19:[1,20],
    20:[22,21],
    21:[7,22],
    22:[24,23],
    23:[11,24]
}

bones_dt_mat = np.array([bones_dt[i] for i in range(24)])

joint_idx_dependents={
    0:[0],
    1:[],
    2:[19,1],
    3:[19,1,2],

    4:[19,3],
    5:[19,3,4],
    6:[19,3,4,5],
    7:[19,3,4,5,6],
    22:[19,3,4,5,6,21],
    21:[19,3,4,5,6,21,20],
    8:[19,7],
    9:[19,7,8],
    10:[19,7,8,9],
    11:[19,7,8,9,10],
    24:[19,7,8,9,10,23],
    23:[19,7,8,9,10,23,22],
    
    12:[0,11],
    13:[0,11,12],
    14:[0,11,12,13],
    15:[0,11,12,13,14],
    16:[0,15],
    17:[0,15,16],
    18:[0,15,16,17],
    19:[0,15,16,17,18],
    20:[19],
}


def get_bones(data_numpy):
    '''
    shape: (c,t,v,m)
    return (c,t,n_bones,m)
    '''
    bones = data_numpy[:,:,bones_dt_mat[:,1],:] - data_numpy[:,:,bones_dt_mat[:,0],:]
    return bones

def get_mask():
    '''
    return (n_joints, n_bones)
    '''
    n_joints=25
    n_bones=24
    m = np.zeros((n_joints,n_bones),float)
    for i in range(n_joints):
        m[i, joint_idx_dependents[i]]=1.0

    return m 

def apply_mask(M, bones):
    '''
    M: (n_joints, n_bones)
    bones: (c,t,n_bones,m)
    '''
    M = M[None,None,None]
    bones = np.transpose(bones,[0,1,3,2])[...,None]

    # (c,t,m,n_joints,1)
    res=np.matmul(M,bones)[...,0]
    res = np.transpose(res,[0,1,3,2])
    return res 

def exchange_limb_list_mat(data_src,data_dst, bone_list):
    n_joints=25
    n_bones=24

    mask_valid = get_mask()

    mask_ex = get_mask()
    if len(bone_list)!=0:
        mask_ex[:,bone_list]=0.0

    mask_ex_inv = 1 - mask_ex 
    mask_ex_inv = mask_valid * mask_ex_inv

    res = apply_mask(mask_ex, get_bones(data_src)) + apply_mask(mask_ex_inv, get_bones(data_dst))

    return res 



def exchange_limb_list_mat_soft(data_src,data_dst, soft_bone_list):
    n_joints=25
    n_bones=24

    soft_bone_list = soft_bone_list.reshape((1,n_bones))

    mask_valid = get_mask()

    mask_ex = get_mask()
    mask_ex = mask_ex * soft_bone_list

    mask_ex_inv = 1 - mask_ex 
    mask_ex_inv = mask_valid * mask_ex_inv

    res = apply_mask(mask_ex, get_bones(data_src)) + apply_mask(mask_ex_inv, get_bones(data_dst))

    return res 


def skeleton_cutmix_multiPeople(data_src, data_dst, p=0.5):
    n_bones=24
    n_drop_num = round(n_bones*p)
    weight_list = np.ones(n_bones)/n_bones
    random_bone_list = np.random.choice(n_bones, n_drop_num, replace=False, p=weight_list)  

    if data_src.shape[-1]==1:          
        x_src_ex_limb_random = exchange_limb_list_mat(data_src, data_dst, random_bone_list)
    elif data_src.shape[-1]==2:
        x_src_ex_limb_random = data_src.copy()
        x_src_ex_limb_random[...,:1] = exchange_limb_list_mat(data_src[...,:1], data_dst[...,:1], random_bone_list)
        
    return x_src_ex_limb_random




def skeleton_cutmix_multiPeople_pis(data_src, data_dst, p=0.5, label=None, dataset=None,T=1.0):


    n_bones=24
    n_drop_num = round(n_bones*p)
    # weight_list = np.ones(n_bones)/n_bones

    if dataset=="n2e":
        weight_list = weight_map_pis[dataset][label]
        weight_list = np.exp(weight_list*T)
        weight_list = weight_list/np.sum(weight_list)
    elif dataset=="e2n":
        weight_list = weight_map_pis[dataset][label]
        weight_list = np.exp(weight_list*T)
        weight_list = weight_list/np.sum(weight_list)
    else:
        raise ValueError()


    random_bone_list = np.random.choice(n_bones, n_drop_num, replace=False, p=weight_list)  

    if data_src.shape[-1]==1:          
        x_src_ex_limb_random = exchange_limb_list_mat(data_src, data_dst, random_bone_list)
    elif data_src.shape[-1]==2:
        x_src_ex_limb_random = data_src.copy()
        x_src_ex_limb_random[...,:1] = exchange_limb_list_mat(data_src[...,:1], data_dst[...,:1], random_bone_list)
        
    return x_src_ex_limb_random




def get_weight_map_pis():

    weight_map_dt = {}

    n_bones=24

    hand_list = [3,4,5,6,20,21,7,8,9,10,22,23]
    foot_list = [11,12,13,14,15,16,17,18]
    hand_torso_list = [0,19,1,2]

    # hand_list = [3,4,5,7,8,9]
    # foot_list = [11,12,13,14,15,16,17,18]
    # hand_torso_list = [0,19,1,2]

    for dataset in ['n2e', 'e2n']:

        if dataset in ["n2e"]:
            w = np.array(n2e_pis_mat)
        elif dataset in ["e2n"]:
            w = np.array(e2n_pis_mat)
        
        else:
            raise ValueError()
        
        n_actions=w.shape[0]
        mat = np.ones((n_actions, n_bones),float)*0.0

        for action_idx in range(n_actions):
            mat[action_idx][hand_list]=w[action_idx][0]
            mat[action_idx][foot_list]=w[action_idx][1]
            mat[action_idx][hand_torso_list]=w[action_idx][2] 

        weight_map_dt[dataset]=mat
    return weight_map_dt



n2e_pis_mat = [[0.994, 0.124, 0.26], [0.891, 0.355, 0.422], [0.995, 0.154, 0.24], [1.013, 0.27, 0.4], [1.012, 0.196, 0.504], [0.977, 0.969, 0.939], [0.94, 0.828, 0.75], [0.897, 0.945, 0.809], [0.964, 0.571, 0.687], [1.0, 0.404, 0.768], [1.103, 0.385, 0.359], [1.004, 0.228, 0.476], [0.96, 0.432, 0.457], [0.963, 0.403, 0.299], [0.794, 0.206, 0.309], [0.91, 0.444, 0.592], [0.996, 0.959, 0.873], [1.004, 0.876, 0.891], [1.004, 0.989, 0.996], [1.0, 0.739, 0.945], [1.0, 0.335, 0.466], [0.996, 0.526, 0.494], [0.927, 0.978, 0.853]]
e2n_pis_mat = [[1.012, 0.607, 0.594], [0.999, 0.549, 0.72], [0.916, 0.71, 0.71], [0.82, 0.804, 0.528], [0.956, 0.678, 0.817], [0.985, 0.957, 0.954], [0.997, 0.903, 0.846], [0.975, 0.906, 0.683], [0.973, 0.798, 0.789], [1.027, 0.5, 0.681], [0.847, 0.598, 0.435], [0.978, 0.216, 0.387], [1.006, 0.312, 0.57], [0.862, 0.34, 0.292], [0.984, 0.467, 0.562], [0.951, 0.365, 0.645], [1.006, 0.879, 0.93], [1.002, 0.912, 0.944], [1.003, 0.967, 0.993], [0.992, 0.92, 0.986], [1.02, 0.18, 0.592], [0.992, 0.57, 0.686], [1.054, 0.755, 1.0]]



weight_map_pis = get_weight_map_pis()