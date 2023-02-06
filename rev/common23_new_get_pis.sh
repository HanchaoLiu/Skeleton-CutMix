
# input ($1=device, $2=config_idx)
# sh rev/common23_new_get_pis.sh 0 0
# sh rev/common23_new_get_pis.sh 1 1

device=$1
seed_list=(0)

config_idx=$2

config_list=("common23New_n2e"
             "common23New_e2n"
             )

test_set_list=(
    "data_path='common23_230201/ntu/test_data_joint_len64.npy',label_path='common23_230201/ntu/test_label.pkl'"
    "data_path='common23_230201/etri/test_data_joint_len64.npy',label_path='common23_230201/etri/test_label.pkl'"
)


use_val_list=(1 1 1)
batch_size_list=(64 64 64)

model_name_list=("agcnThin")
model_list=("nets.agcn.agcn.Model_thin_partLevel")
model=${model_list[0]}
model_name=${model_name_list[0]}

use_aug_list=(1 0 0)
use_aug_name_list=('skcutmix' 'skcutmix' 'skcutmix')
choose_from_list=('src_x_dst' 'src_x_dst' 'src_x_dst')
theta_list=(30 30 0)

num_epoch=25
p=0.5

for seed in 0; do
    for idx in 0; do

        config=${config_list[config_idx]}
        use_aug=${use_aug_list[idx]}
        method=${use_aug_name_list[idx]}
        choose_from=${choose_from_list[idx]}
        theta=${theta_list[idx]}

        work_dir="z2025_${model_name}_${config}_seed${seed}_useAug${use_aug}_${method}_${choose_from}_ep${num_epoch}_v${idx}_p${p}_theta${theta}_testPartLevel"
        

        # tag="foot"
        for tag in "full" "torsoHead" "hand" "foot"; do 
                                                 
        base_dir="/mnt/action/work_dir/actionlib/z2025_agcnPart_${config}_seed0_useAug1_skcutmix_src_x_dst_ep25_v0_p0.5_theta30_${tag}"
        weights="${base_dir}/model_ep24.pt"
        echo "config=${config}, use_val=${use_val_list[idx]}, work_dir=${work_dir}"
        echo "using ${weights}"
       

        python test_model/main_skmix_test.py \
            --work-dir ${work_dir} \
            --config "rev/${config}.yaml" \
            --weights ${weights} --phase 'test' \
            --num-worker 1 \
            --save-interval 1 --eval-interval 1 \
            --model ${model} \
            --num-epoch ${num_epoch}  \
            --seed ${seed} --device ${device} \
            --use-val ${use_val_list[idx]} --batch-size ${batch_size_list[idx]} \
            --train-feeder-args "theta=${theta},p1=${p},p2=${p},method='${method}',choose_from='${choose_from}'" --use-aug ${use_aug}  \
            --weight_aug 0.8 \
            --train-feeder "feeders.feeder.Feeder_SameLabelPair_skcutmix_basic" \
            --step 15 20 \
            --weight_dst_noAug 0.2 \
            --output_name "result_${tag}" \
            --model-args "graph='nets.agcn.graph.ntu_rgb_d_part_v2.Graph_${tag}'" \
            --test-feeder-args ${test_set_list[config_idx]}
        done 
    done
done

