data_type=nerf_synthetic

SCENE=mic
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=chair
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=drums
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=ficus

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=hotdog

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=lego

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=materials

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=ship

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 


