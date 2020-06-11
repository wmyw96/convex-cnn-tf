for d in 0 2 3
do
for t in 40 50 60 70 80 90
do
mkdir glogs-exp2/c10-vgg19-l12-w1-sm-e$t-s$d/
for i in $(seq 2 16)
do
echo $t $i $d
python train_graft.py --gpu 1 --modeldir ../../data/cifar10-models/graft-cifar10.vgg19_nobn_l12_1/sd$d/epoch$t/ --model1dir ../../data/cifar10-models/graft-cifar10.vgg19_nobn_l12_1/sd$d/epoch$t/ --exp_id graft-cifar10.vgg19_nobn_l12_1 --statlogdir glogs-exp2/c10-vgg19-l12-w1-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp
done
done
done
