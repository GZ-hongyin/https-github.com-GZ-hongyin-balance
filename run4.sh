# CUDA_VISIBLE_DEVICES=0 python balanced_mix.py --data_type 'cifar10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/exp' --imb_ratio 0.01  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=0 python balanced_mix.py --data_type 'cifar10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/exp' --imb_ratio 0.02  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=0 python balanced_mix.py --data_type 'cifar10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/exp' --imb_ratio 0.1  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=0 python balanced_mix.py --data_type 'cifar10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/step' --imb_ratio 0.01  --imb_type 'step' --alpha 0.3
# CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/step' --imb_ratio 0.02  --imb_type 'step' --alpha 0.3
# CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar10' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/step' --imb_ratio 0.1  --imb_type 'step' --alpha 0.3


# CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar100' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/exp' --imb_ratio 0.01  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar100' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/exp' --imb_ratio 0.02  --imb_type 'exp' --alpha 0.3
# CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar100' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/exp' --imb_ratio 0.1  --imb_type 'exp' --alpha 0.3
CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar100' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/step' --imb_ratio 0.01  --imb_type 'step' --alpha 0.3
CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar100' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/step' --imb_ratio 0.02  --imb_type 'step' --alpha 0.3
CUDA_VISIBLE_DEVICES=1 python balanced_mix.py --data_type 'cifar100' --loss_type 'CE' --train_rule 'none' --resultFolder 'Results/resnet34/cifar/step' --imb_ratio 0.1  --imb_type 'step' --alpha 0.3
