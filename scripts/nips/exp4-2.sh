for d in 0
do
for t in 1 2 5 10 20
do
mkdir glogs-exp4/c100-vgg16-l2-w1-sm-e$t-s$d/
for i in $(seq 2 13)
do
echo $t $i $d
python train_graft.py --gpu 3 --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l2_1/sd$d/epoch$t/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l2_1/sd$d/epoch$t/ --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs-exp4/c100-vgg16-l2-w1-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp
done
done
done
