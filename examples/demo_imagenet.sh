wget -N http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel -P models/

gpu=0
model_def=models/bvlc_reference_caffenet.prototxt
model_weights=models/bvlc_reference_caffenet.caffemodel
model_name=caffenet
mean='104.00,116.66,122.67'
pad_size=227
eps=4
attack_method=iterative_fgsm

image=data/cat.jpg
out_dir=output
dataset=imagenet

python main.py --gpu ${gpu} --model_def ${model_def} \
--model_weights ${model_weights} --image ${image} --pad_size ${pad_size} \
--eps ${eps} --out_dir ${out_dir} --attack_method ${attack_method} \
--model_name ${model_name} --mean ${mean} --dataset ${dataset}
