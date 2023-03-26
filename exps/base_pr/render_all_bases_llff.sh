data_type=llff

SCENE=fern
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=flower
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=fortress

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=horns

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=leaves

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=orchids

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=room

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=trex

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 
