for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_28/20-02-03-21-37/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l2_28/20-02-03-21-37/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l2_28 --statlogdir logs/c10-vgg16-l2-28/graft_$i.pkl --gpu 0 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l2_28_$i.log
done

