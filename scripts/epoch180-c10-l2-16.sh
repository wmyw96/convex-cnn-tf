for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_16/20-02-03-08-38/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_16/20-02-03-08-38/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l2_16 --statlogdir logs/c10-vgg16-l2-16/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l2_16_$i.log
done
