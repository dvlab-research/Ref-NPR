conda env create -f environment.yml 
conda activate RefNPR
# install customized cuda kernels
mv svox2- svox2
python -m pip install . --upgrade --use-feature=fast-deps
mv svox2 svox2-
python -m pip install gdown
# to avoid package name confliction.