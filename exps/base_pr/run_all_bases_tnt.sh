data_type=tnt

SCENE=Family
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json

SCENE=Horse
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json

SCENE=M60

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json

SCENE=Playground

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json

SCENE=Train

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json

SCENE=Truck

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/tnt.json
