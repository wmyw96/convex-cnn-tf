for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_z5/20-01-31-12-23/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_4/20-01-31-12-23/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_z5 --statlogdir logs/c10-vgg16-l12-4z/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l12_4z_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_4/20-01-31-12-23/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_z5/20-01-31-12-23/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_4 --statlogdir logs/c10-vgg16-l12-z4/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l12_z4_$i.log
done

