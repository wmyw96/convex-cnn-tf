for t in 5 10 20 40 60 120 180
do
mkdir glogs/c100-vgg16-l12-e$t-b-s2/
mkdir glogs/c100-vgg16-l12-e$t-f-s2/
for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch200/ --model1dir  ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch$t/ --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs/c100-vgg16-l12-e$t-f-s2/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >tmp
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch$t/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs/c100-vgg16-l12-e$t-b-s2/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >tmp
done
done

