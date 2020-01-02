for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch180/ --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch50/ --exp_id graft.vgg16_nobn_l12 --gpu 3 --seed 0 --nanase $i >log/vgg16_epoch50_$i.log
done

