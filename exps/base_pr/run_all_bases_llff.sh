data_type=llff

SCENE=fern
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=flower
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=fortress

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=horns

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=leaves

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=orchids

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=room

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json

SCENE=trex

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/llff.json
