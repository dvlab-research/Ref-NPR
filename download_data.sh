# install google drive downloading tool
python -m pip install gdown

mkdir -p data
cd data

# LLFF
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip && mv nerf_llff_data llff

# Synthetic
gdown --fuzzy https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=share_link
unzip nerf_synthetic.zip

# T&T
echo 'For TNT download , please refer to the official site: https://www.tanksandtemples.org/download/'

# testdata
gdown 1VUZSEOxJXoYO_NMDNiQm8fwFMLpkhJEu
mkdir ref_case
unzip ref_case.zip -d ref_case

cd ..
