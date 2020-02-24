
for t in 1 5 8 10 20 30
do
mkdir glogs/c100-vgg16-l12-w1-sm-e$t-s0/
for i in $(seq 2 13)
do
echo $t $i
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch$t/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch$t/ --exp_id graft-cifar100.vgg16_nobn_l12_1 --statlogdir glogs/c100-vgg16-l12-w1-sm-e$t-s0/graft_$i.pkl --logdir mlogs/ --gpu 1 --seed 0 --nanase $i >tmp
done
done


