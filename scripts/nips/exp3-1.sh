for d in 0
do
for t in 1 2 5 10 20 30 40 50 60 70 80 90
do
mkdir glogs-exp3/c100-vgg16-l12-w1-1-2-sm-e$t-s$d/
for i in $(seq 2 13)
do
echo $t $i $d
python train_graft.py --gpu 0 --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1_1_2/sd$d/epoch$t/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1_1_2/sd$d/epoch$t/ --exp_id graft-cifar100.vgg16_nobn_l12_1_1_2 --statlogdir glogs-exp3/c100-vgg16-l12-w1-1-2-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp
done
done
done
