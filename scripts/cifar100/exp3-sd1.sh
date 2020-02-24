for k in 6
do
mkdir glogs/c100-vgg16-l2-w$k-s1/
for i in $(seq 2 13)
do
echo 'exp1' $k $i
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l2_$k/sd1/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l2_$k/sd1/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l2_$k --statlogdir glogs/c100-vgg16-l2-w$k-s1/graft_$i.pkl --logdir mlogs/ --gpu 2 --seed 1 --nanase $i >exp2-sd1-tmp.log
done
done
