for d in 1
do
for t in 1 2 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 200
do
mkdir glogsnew/c10-vgg16-l12-w1-di-sm-e$t-s$d/
for i in $(seq 2 13)
do
echo $t $i $d
python train_graft.py --gpu 1 --modeldir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l12_1_di/sd$d/epoch$t/ --model1dir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l12_1_di/sd$d/epoch$t/ --exp_id graft-cifar10.vgg16_nobn_l12_1_di --statlogdir glogsnew/c10-vgg16-l12-w1-di-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp$d
done
done
done
