
for k in $(seq 2 5)
do
for i in $(seq 2 13)
do
echo $k $i
python train_graft.py --modeldir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_$k/sd0/epoch200/ --model1dir ../../data/cifar100-models/graft-cifar100.vgg16_nobn_l12_1/sd0/epoch200/ --exp_id graft-cifar100.vgg16_nobn_l12_$k --statlogdir glogs/c100-vgg16-l12-w1_$k-s0/graft_$i.pkl --logdir mlogs/ --gpu 0 --seed 0 --nanase $i >tmp
done
done

