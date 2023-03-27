conda env create -f environment.yml 
conda activate RefNPR
# run `python -c "import torch; print(torch.cuda.device_count())"` to examine your torch version
# if you installed a cpu version, manually UNINSTALL and re-install certain gpu versions
# run `conda uninstall pytorch torchvision torchaudio` then run
# e.g. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`

# CUDA required here
# install customized cuda kernels, which takes about 5 minutes to finish
# this trick is to avoid package name confliction
mv svox2- svox2
python -m pip install . --upgrade --use-feature=fast-deps
mv svox2 svox2-
