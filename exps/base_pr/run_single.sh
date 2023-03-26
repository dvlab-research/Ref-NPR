data_type=$1 # style_folder
SCENE=$2 # out_dir

ckpt_svox2=./exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=./data/${data_type}/${SCENE}
python -W ignore opt/opt.py -t ${ckpt_svox2} ${data_dir} \
                -c ./opt/configs/${data_type}.json