
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch8/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e8f/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e8f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch8/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e8b/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e8b_$i.log
done


