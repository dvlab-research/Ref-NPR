# Ref-NPR: Reference-Based Non-Photorealistic Radiance Fields for Controllable Scene Stylization (CVPR2023)

This is the official implementation of the Ref-NPR paper

- **Ref-NPR: Reference-Based Non-Photorealistic Radiance Fields for Controllable Scene Stylization**   
*Yuechen Zhang, Zexin He, Jinbo Xing, Xufeng Yao, Jiaya Jia*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2023  
[ [arXiv](https://arxiv.org/abs/2212.02766) ] [ [Project Page](https://ref-npr.github.io/) ] [ [BibTeX](./assets/bib.txt) ] [ [Video](https://youtu.be/jnsnrTwVSBw) ] [ [Data](https://drive.google.com/drive/folders/1b6L250lrBrSxfKYPmDBHuY_EP9n7WKnA?usp=share_link) ]

Ref-NPR is a powerful tool for single-image scene stylization, allowing users to create stylized versions of scenes based on a given stylized view. The stylized view can be generated through various methods, including hand drawings, 2D image stylization techniques, or text-driven controllable generation (e.g., using ControlNet). Ref-NPR is also compatible with 3D objects represented in NeRF (Neural Radiance Fields) format.


| Sourse Content                  | Hand Drawing<br> Style Image       | ControlNet<br> "Chinese Painting" | ControlNet<br>"Kyoto Animation"  |
|---------|---------|---------|---------|
| <image width="100%" src="assets/style_image_content.png"> |<image  width="100%" src="assets/style_image_colorful.png"> |<image  width="100%" src="assets/style_image_chinese.png"> | <image  width="100%" src="assets/style_image_animation.png"> |
| <image width="100%" autoplay loop muted src="assets/test_renders_path_flower_content.gif"> |<image width="100%" autoplay loop muted src="assets/test_renders_path_flower_colorful.gif"> |<image width="100%" autoplay loop muted src="assets/test_renders_path_flower_chinese.gif"> | <image width="100%" autoplay loop muted src="assets/test_renders_path_flower_animation.gif"> |


For more examples, please refer our gallary in the [project page](https://ref-npr.github.io/).

## Quick start

### 1. Install environment & Data download
```bash
bash ./create_env.sh
bash ./download_data.sh
```

### 2. (Optional) Photo-Realistic training (rendering) on all scenes
You can also utilize your pre-trained Plenoxel models with Ref-NPR. If you prefer not to apply basic PhotoRealistic (PR) models to all scenes, you can skip directly to step 3. This will automatically execute the appropriate PR model for the given scene.
```bash
cd opt
bash ../exps/base_pr/run_all_bases_syn.sh
bash ../exps/base_pr/run_all_bases_llff.sh
bash ../exps/base_pr/run_all_bases_tnt.sh
```

### 3. Run Ref-NPR
```bash
# Ref-NPR: 
# run_single.sh [style folder] [epoch_multi=2] [out_dir=./exps/refnpr]
bash ./exps/refnpr/run_single.sh ./data/ref_case/flower_control/
```
The optimized artistic radiance field is inside ```exps/refnpr/flower_control/exp_out```, while the photorealistic one is inside ```opt/base_pr/ckpt_svox2/[scene_name(flower)]```.

<br/>

**(Optional)** In addition to Ref-NPR, we provide implementations of ARF and SNeRF for comparison purposes. Please note that our SNeRF implementation is not the official version, as it is not open-sourced. We use Gatys' stylization module as a substitute in our SNeRF implementation.
```bash
# ARF: 
# run_single.sh [style folder] [epoch_multi=2] [out_dir=./exps/arf]
bash ./exps/arf/run_single.sh ./data/ref_case/flower_control/

# SNeRF: 
# run_single.sh [style folder] [epoch_multi=1] [out_dir=./exps/snerf]
bash ./exps/snerf/run_single.sh ./data/ref_case/flower_control/
```

### 4. Customization on own data
*Training base on your own scene data:*\
Please follow the steps on [Plenoxel](https://github.com/sxyu/svox2) to prepare your own custom training content data.

*Use your own style reference:*\
If you want to customize your own style example, you can create a reference case follow the format as this:
```bash
├── data
│   ├── ref_case
│   │   ├── ref_case_1
│   │   │   ├── style.png
│   │   │   ├── data_config.json
│   │   ├── ref_case_2
│   │   │   ├── style.png
│   │   │   ├── data_config.json
└── ...
```
Here is an example of ```data_config.json```. Please be noted that one of ```["tmpl_idx_train",
"tmpl_idx_test"]``` should be set to ```null```. 

<br/>

To get content from training set, you can choose the image from training dataset. We provide a simple script to build choose the image and build a config:
```bash
python ./exps/ref_get_example.py
```

<br/>

To get the test sequence content, you can run scripts ```../exps/base_pr/render_all_bases_[dataset_name].sh``` (or part of it for a specific scene). 
```python
{
  "dataset_type": "llff",           # dataset type in ["llff", "synthetic", "tnt"]
  "scene_name": "flower",           # scene name in your used dataset
  "style_img": "./data/ref_case/flower_control/style.png", # path to reference image
  "tmpl_idx_train": null,           # index if you use a *TRAINING* set image as reference.
  "tmpl_idx_test": 112,             # index if you use a *TEST* set image as reference.
  "color_pre": true,                # whether enable the color matching loss [default true]
  "style_name": "flower_control"    # case name for output folders.
}
```

<br/>

For multi-reference case, just use a list to store the style image and indexes:
```python
{
  ...
  "style_img": ["./data/ref_case/playground/style_049.png",
                "./data/ref_case/playground/style_088.png",
                "./data/ref_case/playground/style_157.png"],
  "tmpl_idx_train": [49, 88, 157],
  ...
}
```

<br/>

## Citation
```
@inproceedings{
      zhang2023refnpr,
      title={Ref-{NPR}: Reference-Based Non-Photorealistic Radiance Fields for Controllable Scene Stylization},
      author={Zhang, Yuechen and He, Zexin and Xing, Jinbo and Yao, Xufeng and Jia, Jiaya},
      booktitle={CVPR},
      year={2023}
}
```

## Acknowledgement and References:
This repository is built based on ARF. We would like to thank authors of [Plenoxel](https://github.com/sxyu/svox2) and [ARF](https://github.com/Kai-46/ARF-svox2) for open-sourcing their wonderful implementations.

