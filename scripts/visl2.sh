for i in $(seq 2 8)
do
python results/vis_graftlayer.py --logdir logs/vgg16-l2/graft_$i.pkl --nanase $i
done
