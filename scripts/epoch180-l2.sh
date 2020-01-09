for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l2/20-01-08-02-09/epoch180 --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l2/20-01-08-02-09/epoch180 --exp_id graft.vgg16_nobn_nol --gpu 0 --seed 0 --nanase $i >log/vgg16_epoch180_l2_$i.log
done

