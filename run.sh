#!/usr/bin/env bash


# CUDA_VISIBLE_DEVICES=2 python balanced_mix_cinic.py --data_type 'cinic10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cinic/exp' --imb_ratio 0.01  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=2 python balanced_mix_cinic.py --data_type 'cinic10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34//cinic/exp' --imb_ratio 0.02  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=2 python balanced_mix_cinic.py --data_type 'cinic10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cinic/exp' --imb_ratio 0.1  --imb_type 'exp' --alpha 0.3


CUDA_VISIBLE_DEVICES=2 python balanced_mix_cinic.py --data_type 'cinic10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cinic/step' --imb_ratio 0.01  --imb_type 'step' --alpha 0.3
CUDA_VISIBLE_DEVICES=2 python balanced_mix_cinic.py --data_type 'cinic10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34//cinic/step' --imb_ratio 0.02  --imb_type 'step' --alpha 0.3
CUDA_VISIBLE_DEVICES=2 python balanced_mix_cinic.py --data_type 'cinic10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cinic/step' --imb_ratio 0.1  --imb_type 'step' --alpha 0.3


