for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-10-06/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-10-06/epoch60/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e60f/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e60f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-10-06/epoch60/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-10-06/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e60b/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e60b_$i.log
done

