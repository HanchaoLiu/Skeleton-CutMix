We describe how to prepare data.


1. ntu_gendata.py, etri_gendata.py
Generate train/test data (*.npy), train/test label(*.pkl).
(View alignment)

2. get_length.py, ntu_gendata_length.py, etri_gendata_length.py
Resample to 64 frames. 
(Adaptive temporal sampling)

4. gen_ntu_etri23_split.py 
make pairs and generate paired dataset. Also generate 10-shot target samples.


Then you should have files like

common23_230201/{domain}/train_data_joint_len64.npy
common23_230201/{domain}/train_label.pkl
common23_230201/{domain}/train_10shot_data_joint_len64.npy
common23_230201/{domain}/train_10shot_label.pkl
common23_230201/{domain}/test_data_joint_len64.npy
common23_230201/{domain}/test_label.pkl

domain = ntu/etri



