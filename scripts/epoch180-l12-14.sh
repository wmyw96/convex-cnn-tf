for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l12_1/20-01-30-18-19/epoch180 --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l12_3x/20-01-30-18-04/epoch180 --exp_id graft.vgg16_nobn_l12 --statlogdir logs/vgg16-l12-14/graft_$i.pkl --gpu 1 --seed 0 --nanase $i >log/vgg16_epoch180_l12_14_$i.log
done

