for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch7/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e8f/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e8f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch7/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e8b/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e8b_$i.log
done


for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch19/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e20f/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e20f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch19/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e20b/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e20b_$i.log
done

for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch39/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e40f/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e40f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch39/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e40b/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e40b_$i.log
done


for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch59/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e60f/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e60f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch59/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e60b/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e60b_$i.log
done

