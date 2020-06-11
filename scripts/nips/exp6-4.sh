for d in 0
do
for t in 130 140 150 160 180 200
do
mkdir glogs-exp6/c10-vgg19-l12-w1-di-sm-e$t-s$d/
for i in $(seq 14 16)
do
echo $t $i $d
python train_graft.py --gpu 2 --modeldir ../../data/cifar10-models/graft-cifar10.vgg19_nobn_l12_1_di/sd$d/epoch$t/ --model1dir ../../data/cifar10-models/graft-cifar10.vgg19_nobn_l12_1_di/sd$d/epoch$t/ --exp_id graft-cifar10.vgg19_nobn_l12_1_di --statlogdir glogs-exp6/c10-vgg19-l12-w1-di-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp
done
done
done

