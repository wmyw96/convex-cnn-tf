for d in 0
do
for t in 30 40 50 60 70
do
mkdir glogs-exp5/c100-vgg16-l12-w1-sm-e$t-s$d/
for i in $(seq 2 13)
do
echo $t $i $d
python train_graft.py --gpu 3 --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1_di/sd$d/epoch$t/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1_di/sd$d/epoch$t/ --exp_id graft-cifar100.vgg16_nobn_l12_1_di --statlogdir glogs-exp5/c100-vgg16-l12-w1-sm-e$t-s$d/graft_$i.pkl --logdir mlogs/ --seed $d --nanase $i >tmp
done
done
done

