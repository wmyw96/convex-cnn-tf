for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_4/20-02-02-22-17/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_4/20-02-02-22-17/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l2_4 --statlogdir logs/c10-vgg16-l2-4/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l2_4_$i.log
done

