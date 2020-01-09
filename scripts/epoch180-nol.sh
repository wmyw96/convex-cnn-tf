for i in $(seq 2 13)
do
python train_graft.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_nol/20-01-07-11-58/epoch180 --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_nol/20-01-07-11-58/epoch180 --exp_id graft.vgg16_nobn_nol --gpu 3 --seed 0 --nanase $i >log/vgg16_epoch180_nol_$i.log
done

