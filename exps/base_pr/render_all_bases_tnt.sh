data_type=tnt

SCENE=Family
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=Horse
ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=M60

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=Playground

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=Train

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 

SCENE=Truck

ckpt_svox2=../exps/base_pr/ckpt_svox2/${data_type}/${SCENE}
data_dir=../data/${data_type}/${SCENE}
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path 
python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --train 
