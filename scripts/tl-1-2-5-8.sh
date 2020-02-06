for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch0/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e1f/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e1f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch0/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e1b/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e1b_$i.log
done

for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch1/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e2f/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e2f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch1/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e2b/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e2b_$i.log
done

for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch4/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e5f/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e5f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch4/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch179/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e5b/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e5b_$i.log
done

for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch8/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e8f/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e8f_$i.log
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch8/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_1/20-02-05-18-28/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/tl-c10-vgg16-l12-e8b/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/tl_c10_vgg16_l12_e8b_$i.log
done

