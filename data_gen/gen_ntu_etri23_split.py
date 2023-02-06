import os,sys
import numpy as np 
from IPython import embed 

import pickle

ntu_actions={
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up (from sitting position)",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear a shoe",
    17: "take off a shoe",
    18: "wear on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping (one foot jumping)",
    27: "jump up",
    28: "make a phone call/answer phone",
    29: "playing with phone/tablet",
    30: "typing on a keyboard",
    31: "pointing to something with finger",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands together",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put the palms together",
    40: "cross hands in front (say stop)",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head (headache)",
    45: "touch chest (stomachache/heart pain)",
    46: "touch back (backache)",
    47: "touch neck (neckache)",
    48: "nausea or vomiting condition",
    49: "use a fan (with hand or paper)/feeling warm",
    50: "punching/slapping other person",
    51: "kicking other person",
    52: "pushing other person",
    53: "pat on back of other person",
    54: "point finger at the other person",
    55: "hugging other person",
    56: "giving something to other person",
    57: "touch other person's pocket",
    58: "handshaking",
    59: "walking towards each other",
    60: "walking apart from each other",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at the basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggling table tennis balls",
    67: "hush (quite)",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make ok sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper (using scissors)",
    77: "snapping fingers",
    78: "open bottle",
    79: "sniff (smell)",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand back",
    87: "put on bag",
    88: "take off bag",
    89: "put something into a bag",
    90: "take something out of a bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "hands up (both hands)",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "running on the spot",
    100: "butt kicks (kick backward)",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit other person with something",
    107: "wield knife towards other person",
    108: "knock over other person (hit with body)",
    109: "grab other person's stuff",
    110: "shoot at other person with a gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry something with other person",
    115: "take a photo of other person",
    116: "follow other person",
    117: "whisper in other person's ear",
    118: "exchange things with other person",
    119: "support somebody with hand",
    120: "finger-guessing game (playing rock-paper-scissors)",
}
ntu60_actions={k:v for k,v in ntu_actions.items() if k <= 60}


etri_actions = {
    1: "eating food with a fork",
    2: "pouring water into a cup",
    3: "taking medicine",
    4: "drinking water",
    5: "putting food in the fridge/taking food from the fridge",
    6: "trimming vegetables",
    7: "peeling fruit",
    8: "using a gas stove",
    9: "cutting vegetable on the cutting board",
    10: "brushing teeth",
    11: "washing hands",
    12: "washing face",
    13: "wiping face with a towel",
    14: "putting on cosmetics",
    15: "putting on lipstick",
    16: "brushing hair",
    17: "blow drying hair",
    18: "putting on a jacket",
    19: "taking off a jacket",
    20: "putting on/taking off shoes",
    21: "putting on/taking off glasses",
    22: "washing the dishes",
    23: "vacuuming the floor",
    24: "scrubbing the floor with a rag",
    25: "wiping off the dinning table",
    26: "rubbing up furniture",
    27: "spreading bedding/folding bedding",
    28: "washing a towel by hands",
    29: "hanging out laundry",
    30: "looking around for something",
    31: "using a remote control",
    32: "reading a book",
    33: "reading a newspaper",
    34: "handwriting",
    35: "talking on the phone",
    36: "playing with a mobile phone",
    37: "using a computer",
    38: "smoking",
    39: "clapping",
    40: "rubbing face with hands",
    41: "doing freehand exercise",
    42: "doing neck roll exercise",
    43: "massaging a shoulder oneself",
    44: "taking a bow",
    45: "talking to each other",
    46: "handshaking",
    47: "hugging each other",
    48: "fighting each other",
    49: "waving a hand",
    50: "flapping a hand up and down (beckoning)",
    51: "pointing with a finger",
    52: "opening the door and walking in",
    53: "fallen on the floor",
    54: "sitting up/standing up",
    55: "lying down",
}
etri_action_inv={v:k for k,v in etri_actions.items()}


# ntu120_list=[ntu_actions[i+1] for i in range(120)]
ntu120_list=[ntu60_actions[i+1] for i in range(60)]
etri55_list=[etri_actions[i+1] for i in range(55)]

print(ntu120_list)
print(etri55_list)


ntu23_dt = {
0 : ['eat meal/snack'] ,
1 : ['drink water'] ,
2 : ['brushing teeth'] ,
3 : ['rub two hands together'] ,
4 : ['brushing hair'] ,
5 : ['wear jacket'] , 
6 : ['take off jacket'] , 
7 : ['wear a shoe', 'take off a shoe'] , 
8 : ['wear on glasses', 'take off glasses'] , 
9 : ['reading'] , 
10 : ['writing'] , 
11 : ['make a phone call/answer phone'] , 
12 : ['playing with phone/tablet'] ,
13 : ['typing on a keyboard'] , 
14 : ['clapping'] , 
15 : ['wipe face'] , 
16 : ['nod head/bow'] , 
# 17 : ['shake head'] , 
17 : ['handshaking'] , 
18 : ['hugging other person'] ,
19 : ['punching/slapping other person'] , 
20 : ['hand waving'] , 
21 : ['pointing to something with finger'] ,
22 : ['falling'] ,
}

etri23_dt = {
    0 : ['eating food with a fork'] , 
    1 : ['drinking water'] , 
    2 : ['brushing teeth'] , 
    3 : ['washing hands'] , 
    4 : ['brushing hair'] , 
    5 : ['putting on a jacket'] , 
    6 : ['taking off a jacket'] , 
    7 : ['putting on/taking off shoes'] , 
    8 : ['putting on/taking off glasses'] , 
    9 : ['reading a book'] , 
    10 : ['handwriting'] , 
    11 : ['talking on the phone'] , 
    12 : ['playing with a mobile phone'] , 
    13 : ['using a computer'] , 
    14 : ['clapping'] , 
    15 : ['rubbing face with hands'] , 
    16 : ['taking a bow'] ,
    17 : ['handshaking'] , 
    18 : ['hugging each other'] , 
    19 : ['fighting each other'] , 
    20 : ['waving a hand'] , 
    21 : ['pointing with a finger'] , 
    22 : ['fallen on the floor'] , 
}

def get_inv_action_map(map_dt):
    '''
    map_dt: {action_idx: [j1, j2, ...]}
    '''
    inv_map_dt = {}
    for action_idx, v_list in map_dt.items():
        for v in v_list:
            inv_map_dt[v]=action_idx
    return inv_map_dt

    

def load_pkl(path):
    return pickle.load(open(path,"rb"))


def write_pkl(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def main():

    split23_name_list = """
1. eat
2. drink
3. brush teeth
4. wash hands
5. brush hair
6. wear clothes
7. take off clothes
8. put on/take off shoes
9. put on/take off glasses
10. read
11. write
12. phone call
13. play with phone
14. use computer
15. clap
16. rub face
17. bow
18. handshake
19. hug
20. fight
21. hand wave
22. point finger
23. fall down
"""

    split23_name_list = split23_name_list.split("\n")
    split23_name_list = [i for i in split23_name_list if len(i)!=0]
    print(split23_name_list)

    for i in range(23):
        print(split23_name_list[i], ntu23_dt[i], etri23_dt[i])

    ntu23_action_idx_dt = {}
    for i in range(23):
        v_list = ntu23_dt[i]
        ntu23_action_idx_dt[i] = [ntu120_list.index(v) for v in v_list]


    etri23_action_idx_dt = {}
    for i in range(23):
        v_list = etri23_dt[i]
        etri23_action_idx_dt[i] = [etri55_list.index(v) for v in v_list]

    print(ntu23_action_idx_dt)
    print(etri23_action_idx_dt)

    ntu23_action_idx_dt_inv = get_inv_action_map(ntu23_action_idx_dt)
    etri23_action_idx_dt_inv= get_inv_action_map(etri23_action_idx_dt)

    
    dataset='ntu'
    split='train'

    ntu_dir = "/home/yqs/liuhc/action/data/ntu220502_resample64/xsub"
    etri_dir = "/home/yqs/liuhc/action/data/etri220501_resample64/adult_elder"

    path_dt = {
        'ntu': {
            'train': ntu_dir + "/train_label.pkl",
            'test':  ntu_dir + "/val_label.pkl",
        },
        'etri': {
            'train': etri_dir + '/train_label.pkl',
            'test': etri_dir + '/test_label.pkl'

        }
    }

    action_map_dt = {
        'ntu': ntu23_action_idx_dt,
        'etri': etri23_action_idx_dt
    }

    inv_action_map_dt = {
        'ntu': ntu23_action_idx_dt_inv, 
        'etri': etri23_action_idx_dt_inv
    }

    # embed()
    # sys.exit(0)



    res_dt = {}
    res_dt['action'] = split23_name_list
    for dataset in ['ntu', 'etri']:
        for split in ['train', 'test']:
            res=get_split23_perclass_stats(path_dt[dataset][split], action_map_dt[dataset])
            res_dt[dataset+"_"+split] = res 

    
    import pandas as pd 
    df = pd.DataFrame(res_dt)
    df.loc['sum'] = df.sum()
    print(df)

    res_dt = {}
    for dataset in ['ntu', 'etri']:
        for split in ['train', 'test']:
            res=get_split23_index_list(path_dt[dataset][split], action_map_dt[dataset])
            res_dt[dataset+"_"+split] = res
    for k,v in res_dt.items():
        print(k,len(v))

    for k,v in res_dt.items():
        assert len(v)==len(np.unique(v))

    selected_map = res_dt

    # etri18_label="/home/yqs/liuhc/action/data/etri220501_resample64/etri18/adult/train_label.pkl"
    # a,b=load_pkl(etri18_label)
    # b=np.array(b)
    # b=pd.Series(b)
    # print(b.value_counts())

    save_idx_path='/home/yqs/liuhc/action/data/common23_230201/stats/common23_selected_idx.pkl'
    write_pkl(save_idx_path, res_dt)


    # make sure to change the action label to [0,22]
    ntu_dir = "/home/yqs/liuhc/action/data/ntu220502_resample64/xsub"
    etri_dir = "/home/yqs/liuhc/action/data/etri220501_resample64/adult_elder"

    output_dir="/home/yqs/liuhc/action/data/common23_230201"

    # ntu
    # ntu_dir + "/val_data.npy", ntu_dir + "/val_label.pkl",

    # ntu train
    save_common23(
        ntu_dir + "/train_data_joint_len64.npy", 
        ntu_dir + "/train_label.pkl",
        output_dir + '/ntu/train_data_joint_len64.npy', 
        output_dir + '/ntu/train_label.pkl', 
        selected_map['ntu_train'], inv_action_map_dt['ntu']
    )

    # ntu test
    save_common23(
        ntu_dir + "/val_data_joint_len64.npy", 
        ntu_dir + "/val_label.pkl",
        output_dir + '/ntu/test_data_joint_len64.npy', 
        output_dir + '/ntu/test_label.pkl', 
        selected_map['ntu_test'], inv_action_map_dt['ntu']
    )

    # etri train
    save_common23(
        etri_dir + "/train_data_joint_len64.npy", 
        etri_dir + "/train_label.pkl",
        output_dir + '/etri/train_data_joint_len64.npy', 
        output_dir + '/etri/train_label.pkl', 
        selected_map['etri_train'], inv_action_map_dt['etri']
    )

    # etri test
    save_common23(
        etri_dir + "/test_data_joint_len64.npy", 
        etri_dir + "/test_label.pkl",
        output_dir + '/etri/test_data_joint_len64.npy', 
        output_dir + '/etri/test_label.pkl', 
        selected_map['etri_test'], inv_action_map_dt['etri']
    )

def gen_common23_10shot():
    output_dir="/home/yqs/liuhc/action/data/common23_230201"
    np.random.seed(0)
    dataset_list = ['ntu', 'etri']
    for dataset in dataset_list:
        print("-"*60)
        data_file=output_dir+f"/{dataset}/train_data_joint_len64.npy"
        label_file=output_dir+f"/{dataset}/train_label.pkl"

        output_data_file=output_dir+f"/{dataset}/train_10shot_data_joint_len64.npy"
        output_label_file=output_dir+f"/{dataset}/train_10shot_label.pkl"

        print(data_file)
        print(label_file)
        print("->", output_data_file)
        print("->", output_label_file)

        # generate a index list of 10 shot for each class.
        selected_10shot_idx_list=[]
        label_all=load_pkl(label_file)
        samples,labels = label_all
        labels=np.array(labels)
        all_actions=np.unique(labels)
        assert np.all(all_actions==np.arange(23))

        
        for action_idx in range(23):
            action_idx_list = np.where(labels==action_idx)[0]
            action_idx_selected_list = np.random.permutation(action_idx_list)[:10]

            selected_10shot_idx_list.append( action_idx_selected_list )
        selected_10shot_idx_list = np.concatenate(selected_10shot_idx_list,0)
        print(selected_10shot_idx_list)
        selected_10shot_idx_list = selected_10shot_idx_list.tolist()

        # slice data and label
        data = np.load(data_file)
        new_data = data[selected_10shot_idx_list]
        new_label = get_label_slice(label_all, selected_10shot_idx_list)
        # print(new_label)
        
        # save to dir
        np.save( output_data_file, new_data)
        write_pkl( output_label_file, new_label )




def get_label_slice(label_all, selected_10shot_idx_list):
    a,b = label_all
    a2 = [a[i] for i in selected_10shot_idx_list]
    b2 = [b[i] for i in selected_10shot_idx_list]
    new_label = [a2,b2]
    return new_label



def save_common23(input_data, input_label, output_data, output_label, selected_list, inv_map_dt):
    print("-"*60)
    print("input data: ", input_data)
    print("input label:", input_label)
    print("output data: ", output_data)
    print("output label:", output_label)

    data=np.load(input_data)
    samples,labels=load_pkl(input_label)

    data_new=data[selected_list]
    samples_new=[samples[i] for i in selected_list]
    labels_new =[  inv_map_dt[labels[i]] for i in selected_list ]
    assert np.all(np.unique(labels_new)==np.arange(23))
    label_new = [samples_new, labels_new]

    np.save(output_data, data_new)
    write_pkl(output_label, label_new)




def get_split23_perclass_stats(label_file, dt):
    res=[]
    samples, labels = load_pkl(label_file)
    labels = np.array(labels)
    for action_idx in range(23):
        action_list=dt[action_idx]
        res_idx=0
        for v in action_list:
            res_idx = res_idx + (labels==v).sum()
        res.append(res_idx)
    return res 


def get_split23_index_list(label_file, dt):
    res=[]
    samples, labels = load_pkl(label_file)
    labels = np.array(labels)
    for action_idx in range(23):
        action_list=dt[action_idx]
        for v in action_list:
            res_idx=np.where(labels==v)[0]
            res.append(res_idx)
    res = np.concatenate(res,0)
    return res 

        
    


if __name__ == "__main__":
    main()
    gen_common23_10shot()