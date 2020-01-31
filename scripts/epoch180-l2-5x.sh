for i in $(seq 12 12)
do
python train_graft.py --statlogdir logs/vgg16-l2-5/graft_$i.pkl --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l2_5x/20-01-20-10-52/epoch180/ --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l2_5x/20-01-20-10-52/epoch180/ --exp_id graft.vgg16_nobn_l2_5x --gpu 0 --seed 0 --nanase $i >log/vgg16_epoch180_l2_5x_$i.log
done

