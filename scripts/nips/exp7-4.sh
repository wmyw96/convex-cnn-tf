for d in 0
do
for t in 70 60 50 40 30 20 10 5 2 1
do
mkdir glogs-exp7/c100-vgg16-l12-w1-di-2s-sm-e$t-s$d/
for i in 10
do
echo $t $i $d
python train_graft.py --gpu 1 --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1_di_2s/sd0_20-06-11-09-29/epoch$t/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1_di_2s/sd0_20-06-11-09-29/epoch$t/ --exp_id graft-cifar100.vgg16_nobn_l12_1_di_2s --statlogdir glogs-exp7/c100-vgg16-l12-w1-di-2s-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp
done
done
done
