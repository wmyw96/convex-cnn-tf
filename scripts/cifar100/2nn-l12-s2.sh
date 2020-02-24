for w in 1 4 6 8
do
python train_2nn.py --seed 2 --gpu 1 --exp_id graft-cifar100.vgg16_nobn_l2_$w --modeldir ../../data/cifar100-models/ --logdir ../../data/cifar100-logs
done

