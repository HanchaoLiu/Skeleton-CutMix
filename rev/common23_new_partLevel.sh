
# input ($1=device $2=part)
# sh rev/common23_new_partLevel.sh 0 0
# sh rev/common23_new_partLevel.sh 1 1
# sh rev/common23_new_partLevel.sh 2 2
# sh rev/common23_new_partLevel.sh 3 3

device=$1
seed_list=(0)

config_list=("common23New_n2e"
             "common23New_e2n"
             )

use_val_list=(1 1 1)
batch_size_list=(64 64 64)

model_name_list=("agcnPart")
model_list=("nets.agcn.agcn.Model_thin_partLevel")
model=${model_list[0]}
model_name=${model_name_list[0]}

use_aug_list=(1 0 0)
use_aug_name_list=('skcutmix' 'skcutmix' 'skcutmix')
choose_from_list=('src_x_dst' 'src_x_dst' 'src_x_dst')
theta_list=(30 30 0)

num_epoch=25
p=0.5

part_list=("hand" "foot" "torsoHead" "full")

for seed in 0; do
    for idx in 0; do

        part=${part_list[$2]}
        config=${config_list[0]}
        use_aug=${use_aug_list[idx]}
        method=${use_aug_name_list[idx]}
        choose_from=${choose_from_list[idx]}
        theta=${theta_list[idx]}

        work_dir="z2025_${model_name}_${config}_seed${seed}_useAug${use_aug}_${method}_${choose_from}_ep${num_epoch}_v${idx}_p${p}_theta${theta}_${part}"
        echo "config=${config}, use_val=${use_val_list[idx]}, work_dir=${work_dir}"

        python rev/main_skmix_save_last.py \
            --work-dir ${work_dir} \
            --config "rev/${config}.yaml" \
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
            --weight_dst_noAug 0.2  \
            --model-args "graph='nets.agcn.graph.ntu_rgb_d_part_v2.Graph_${part}'"
    # done
    done
done

