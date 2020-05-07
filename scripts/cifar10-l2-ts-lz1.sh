for t in 5 10 20 30 50 100 180
do
mkdir glogs/c10-vgg16-l2-w1-lz1-sm-e$t-s0/
for i in $(seq 2 13)
do
echo $t $i
python train_graft.py --modeldir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l2_1_lz1/sd0/epoch$t/ --model1dir ../../data/cifar10-models/graft-cifar10.vgg16_nobn_l2_1_lz1/sd0/epoch$t/ --exp_id graft-cifar10.vgg16_nobn_l2_1_lz1 --statlogdir glogs/c10-vgg16-l2-w1-lz1-sm-e$t-s0/graft_$i.pkl --logdir mlogs/ --gpu 1 --seed 0 --nanase $i >tmp
done
done

