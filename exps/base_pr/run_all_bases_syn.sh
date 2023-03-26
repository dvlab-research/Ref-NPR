data_type=nerf_synthetic

SCENE=chair
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=drums
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=ficus

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=hotdog

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=lego

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=materials

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=ship

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json

SCENE=mic

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python opt.py -t ${ckpt_svox2} ${data_dir} \
                -c configs/nerf_synthetic.json
