import os,sys
import numpy as np 
from IPython import embed 


n_common23=23


def get_perclass_acc(pred, gt, n_actions):
    acc_list = []
    for a in range(n_actions):
        acc = (pred[gt==a]==gt[gt==a]).mean()
        acc_list.append(acc)
    return acc_list


def get23_list():
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
    # print(split23_name_list)
    split23_name_list = [i.replace(" ","").replace("/","_") for i in split23_name_list]
    return split23_name_list

def main_n2e():
    
    n2e_base_dir="/mnt/action/work_dir/actionlib/z2024_agcnThin_common23New_n2e_seed0_useAug1_skcutmix_src_x_dst_ep25_v0_p0.5_theta30_testPartLevel"
    part_list = ["torsoHead", "hand", "foot", "full"]

    dt = {}
    for part in part_list:
        data_path = os.path.join(n2e_base_dir, f"result_{part}.npy")
        x = np.load(data_path)

        dt[part] = x

    dt_res = {}
    for part in part_list:
        x = dt[part]
        dt_res[part] = get_perclass_acc( x[:,0] , x[:,1], n_common23 )

    for part in part_list:
        dt_res[part] = np.array(dt_res[part]) / np.array(dt_res['full'])
    
    import pandas as pd 
    
    pd.set_option("display.precision", 2)
    df = pd.DataFrame(dt_res)
    action_name_list = get23_list()
    df.index = action_name_list
    print(df)
        
    df2 = df[['hand', 'foot', 'torsoHead']]
    print(df2)

    values = df2.values
    values = np.round(values, 3)
    print(values.tolist())

    

def main_e2n():
    
    n2e_base_dir="/mnt/action/work_dir/actionlib/z2024_agcnThin_common23New_e2n_seed0_useAug1_skcutmix_src_x_dst_ep25_v0_p0.5_theta30_testPartLevel"
    part_list = ["torsoHead", "hand", "foot", "full"]

    dt = {}
    for part in part_list:
        data_path = os.path.join(n2e_base_dir, f"result_{part}.npy")
        x = np.load(data_path)
        print(x.shape)
        dt[part] = x

    dt_res = {}
    for part in part_list:
        x = dt[part]
        dt_res[part] = get_perclass_acc( x[:,0] , x[:,1], n_common23 )

    for part in part_list:
        dt_res[part] = np.array(dt_res[part]) / np.array(dt_res['full'])
    
    import pandas as pd 
    
    pd.set_option("display.precision", 2)
    df = pd.DataFrame(dt_res)
    action_name_list = get23_list()
    df.index = action_name_list
    print(df)
        
    df2 = df[['hand', 'foot', 'torsoHead']]
    print(df2)

    values = df2.values
    values = np.round(values, 3)
    print(values.tolist())




if __name__ == "__main__":
    # main()

    # main_n2e()
    
    main_e2n()