
for k in $(seq 2 4)
do
mkdir glogs/c100-vgg16-l12-w$k-1-s2/
for i in $(seq 2 13)
do
echo 'exp2' $k $i
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd2/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_$k/sd2/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l12_$k --statlogdir glogs/c100-vgg16-l12-w$k-1-s2/graft_$i.pkl --logdir mlogs/ --gpu 0 --seed 1 --nanase $i >tmpsd2.log
done
done

