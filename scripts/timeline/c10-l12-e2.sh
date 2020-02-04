
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch2/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e2f/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e2f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch2/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e2b/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e2b_$i.log
done

