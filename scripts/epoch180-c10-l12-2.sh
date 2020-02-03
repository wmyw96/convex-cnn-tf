for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_2/20-01-31-12-22/epoch180 --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_2/20-01-31-12-22/epoch180 --exp_id graft-cifar10.vgg16_nobn_l12_2 --statlogdir logs/c10-vgg16-l12-2/graft_$i.pkl --gpu 2 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l12_2_$i.log
done

