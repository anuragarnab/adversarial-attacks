wget -N http://www.robots.ox.ac.uk/~aarnab/projects/cvpr_2018/models/fcn8_voc.caffemodel -P models/

gpu=0
model_def=models/fcn8_voc.prototxt
model_weights=models/fcn8_voc.caffemodel
model_name=fcn8
mean='103.939,116.779,123.68' # BGR order
pad_size=513
eps=8
attack_method=iterative_fgsm_ll

image=data/2007_000042.jpg
out_dir=output

python main.py --gpu ${gpu} --model_def ${model_def} \
--model_weights ${model_weights} --image ${image} --pad_size ${pad_size} \
--eps ${eps} --is_seg --out_dir ${out_dir} --attack_method ${attack_method} \
--model_name ${model_name} --mean ${mean}