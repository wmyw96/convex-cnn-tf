

for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l2_8/sd0/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l2_8/sd0/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l2_8 --statlogdir glogs/c100-vgg16-l2-w8-s0/graft_$i.pkl --logdir mlogs/ --gpu 1 --seed 0 --nanase $i >tmp
done

