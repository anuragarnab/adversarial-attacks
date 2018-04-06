wget -N http://vobj.cs.princeton.edu/models/dilation10_cityscapes.caffemodel -P models/

gpu=0
model_def=models/dilation10_cityscapes.prototxt
model_weights=models/dilation10_cityscapes.caffemodel
model_name=dilated_context
eps=4
attack_method=iterative_fgsm_ll

pad_size='1396,884'
resize_dims='1024,512'
dataset='cityscapes'
mean='72.39,82.91,73.16' # In BGR order

image=data/frankfurt_000000_000294_leftImg8bit.png
out_dir=output

python main.py --gpu ${gpu} --model_def ${model_def} --model_weights ${model_weights} \
--image ${image} --pad_size ${pad_size} --resize_dims ${resize_dims} \
--eps ${eps} --is_seg --out_dir ${out_dir} --attack_method ${attack_method} \
--dataset ${dataset}  --mean ${mean} --model_name ${model_name} --is_dilated