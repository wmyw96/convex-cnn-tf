for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_10/20-02-02-13-22/epoch180/ --model1dir ../../data/cifar-100-models/graft-cifar10.vgg16_nobn_l12_10/20-02-02-13-22/epoch180/ --exp_id graft-cifar10.vgg16_nobn_l12_1 --statlogdir logs/c10-vgg16-l12-10/graft_$i.pkl --gpu 3 --seed 0 --nanase $i >log/c10_vgg16_epoch180_l12_10_$i.log
done



