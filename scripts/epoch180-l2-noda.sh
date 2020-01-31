for i in $(seq 2 13)
do
python train_graft.py --statlogdir logs/vgg16-l2-noda/graft_$i.pkl --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l2_noda/20-01-16-19-35/epoch180 --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l2_noda/20-01-16-19-35/epoch180 --exp_id graft.vgg16_nobn_l2 --gpu 3 --seed 0 --nanase $i >log/vgg16_epoch180_l2_noda_$i.log
done

