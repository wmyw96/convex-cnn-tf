for d in 0
do
for t in 5 10 30 50 80 120 180
do
mkdir glogsnew/c10-vgg16-l2-w1-di-sm-e$t-s$d/
for i in $(seq 2 13)
do
echo $t $i $d
python train_graft.py --gpu 0 --modeldir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l2_1_di/sd$d/epoch$t/ --model1dir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l2_1_di/sd$d/epoch$t/ --exp_id graft-cifar10.vgg16_nobn_l2_1_di --statlogdir glogsnew/c10-vgg16-l2-w1-di-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp$d
done
done
done
