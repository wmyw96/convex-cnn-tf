
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_5/sd0/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_5/sd0/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l12_5 --statlogdir glogs/c100-vgg16-l12-w5-s0/graft_$i.pkl --logdir mlogs/ --gpu 1 --seed 0 --nanase $i
done
