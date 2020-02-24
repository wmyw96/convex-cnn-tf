

for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch60/ --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs/c100-vgg16-l12-e60f-s0/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >tmp2
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch60/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs/c100-vgg16-l12-e60b-s0/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >tmp2
done

