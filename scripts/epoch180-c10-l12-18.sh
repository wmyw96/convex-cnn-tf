for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_8/20-02-01-20-46/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-01-31-12-17/epoch180 --exp_id graft-cifar10.vgg16_nobn_l12_8 --statlogdir logs/c10-vgg16-l12-18/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l12_18_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-01-31-12-17/epoch180 --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_8/20-02-01-20-46/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/c10-vgg16-l12-81/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l12_81_$i.log
done

