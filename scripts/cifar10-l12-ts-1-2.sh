for t in 30 40 120 150
do
mkdir glogsnew/c10-vgg16-l12-w1-1-2-sm-e$t-s0/
for i in $(seq 2 13)
do
echo $t $i
python train_graft.py --modeldir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l12_1_1_2/sd0_20-05-07-21-04/epoch$t/ --model1dir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l12_1_1_2/sd0_20-05-07-21-04/epoch$t/ --exp_id graft-cifar10.vgg16_nobn_l12_1_1_2 --statlogdir glogsnew/c10-vgg16-l12-w1-1-2-sm-e$t-s0/graft_$i.pkl --logdir mlogs/ --gpu 2 --seed 0 --nanase $i >tmp
done
done

