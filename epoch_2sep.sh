for i in $(seq 5 6)
do
let jmin=$i+1
for j in $(seq $jmin 13)
do
echo $i $j
python train_graft2.py --modeldir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch180/ --model1dir ../../data/cifar-100-models/graft.vgg16_nobn_l12/20-01-01-14-07/epoch180/ --exp_id graft.vgg16_nobn_l12 --gpu 0 --seed 0 --pnanase $i --nanase $j >log/vgg16_sep_$i,$j.log
done
done

