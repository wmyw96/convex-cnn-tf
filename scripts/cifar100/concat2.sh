

for i in $(seq 2 13)
do
python train_graft3.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-01-31-12-17/epoch180 --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch200/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir glogs/c100-c10-l12-s2/graft_$i.pkl --logdir mlogs/ --gpu 1 --seed 0 --nanase $i >ttmp2
done


