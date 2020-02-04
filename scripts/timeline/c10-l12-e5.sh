
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch5/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e5f/graft_$i.pkl --gpu 2 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e5f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch5/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-04-09-47/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e5b/graft_$i.pkl --gpu 2 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e5b_$i.log
done

