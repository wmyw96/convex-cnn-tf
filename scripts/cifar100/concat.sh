
for i in $(seq 2 13)
do
python train_graft3.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch200/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-01-31-12-17/epoch180 --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs/c10-c100-l12-sd2/graft_$i.pkl --logdir mlogs/ --gpu 2 --seed 0 --nanase $i >ttmp
done

