for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch180/ --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l12_1/20-01-30-18-19/epoch180/ --statlogdir logs/vgg16-l12-12/graft_$i.pkl --exp_id graft.vgg16_nobn_l12 --gpu 0 --seed 0 --nanase $i >log/vgg16_epoch180_l12_12_$i.log
done

