# sh rev/common23_new_contrastive_posneg.sh 1 'ccsa'
# sh rev/common23_new_contrastive_posneg.sh 0 'dage'
# Feeder_RandomLabelPair

device=$1
seed_list=(0)

config_list=("common23New_n2e"
             "common23New_e2n"
             )



model_name_list=("agcnThin")
model_list=("nets.agcn.agcn.Model_thin")
model=${model_list[0]}
model_name=${model_name_list[0]}

use_val_list=(1 1 1)
batch_size_list=(64 64 64)
use_aug_list=(1 0 0)
use_aug_name_list=('skcutmix' 'skcutmix' 'skcutmix')
choose_from_list=('src_x_dst' 'src_x_dst' 'src_x_dst')
theta_list=(30 30 0)

num_epoch=25
p=0.5

weight_dst=0.2
weight_contrastive=0.3
margin=1.0
# loss_type='ccsa'
loss_type=$2

for seed in 0; do
    for config_idx in 0; do 
    config=${config_list[config_idx]}

    for idx in 0; do
    
        use_aug=${use_aug_list[idx]}
        method=${use_aug_name_list[idx]}
        choose_from=${choose_from_list[idx]}
        theta=${theta_list[idx]}

        work_dir="z2025_common23New_${config}_idx${idx}_${loss_type}_wdst${weight_dst}_wcon${weight_contrastive}_posneg"
        echo "config=${config}, use_val=${use_val_list[idx]}, work_dir=${work_dir}"

        python rev/main_rev_contrastive.py \
            --work-dir ${work_dir} \
            --config "rev/${config}.yaml" \
            --num-worker 1 \
            --save-interval 1 --eval-interval 1 \
            --model ${model} \
            --num-epoch ${num_epoch}  \
            --seed ${seed} --device ${device} \
            --use-val ${use_val_list[idx]} --batch-size ${batch_size_list[idx]} \
            --train-feeder-args "theta=${theta},choose_from='${choose_from}',use_mmap=0" --use-aug ${use_aug}  \
            --step 15 20 \
            --train-feeder "feeders.feeder.Feeder_PosNegRatioPair" \
            --weight_dst ${weight_dst} --weight_contrastive ${weight_contrastive} --margin ${margin} --loss_type ${loss_type} 
    done
    done 
done
