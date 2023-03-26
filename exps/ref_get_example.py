import os
import cv2
import json
import numpy as np

configs = {
    "dataset": "tnt",   # dataset name
    "scene": "Playground",   # scene name
    "train_idx": 200,    
    # training index, -1 means random choosen. A string name means a choosen image.
    "dst": "data/ref_case/",
    "style_name": "test"          
    # path to save the generated example
}

fail = False
train_idx = -1
dataset = configs['dataset']
scene = configs['scene']

if dataset == 'nerf_synthetic':
    data_path = f'./data/nerf_synthetic/{scene}/train/'
    pathes = sorted(os.listdir(data_path))
    if isinstance(configs["train_idx"], str): # r_K.png
        tmpl_idx_train = pathes.index(configs["train_idx"])
        content_path = os.path.join(data_path, configs["train_idx"])
    if configs["train_idx"] == -1:
        configs["train_idx"] = np.random.randint(0, len(pathes))
    
    if configs["train_idx"] >= len(pathes):
        fail=True
    else:
        content_path = os.path.join(data_path, pathes[configs["train_idx"]])
    tmpl_idx_train = configs["train_idx"]
    
if dataset == 'llff':
    data_path = f'./data/llff/{scene}/images_4/'
    pathes = sorted(os.listdir(data_path))

    for i in range(len(pathes)):
        if i % 8 == 0:
            pathes[i] = -1
    pathes = sorted([i for i in pathes if i != -1])
    
    if isinstance(configs["train_idx"], str): # imageK.png
        tmpl_idx_train = pathes.index(configs["train_idx"])    
        content_path = os.path.join(data_path, configs["train_idx"])
            
    if configs["train_idx"] == -1:
        configs["train_idx"] = np.random.randint(0, len(pathes))
    
    if configs["train_idx"] >= len(pathes):
        fail=True
    else:
        content_path = os.path.join(data_path, pathes[configs["train_idx"]])
    tmpl_idx_train = configs["train_idx"]
    
if dataset == 'tnt':
    data_path = f'./data/tnt/{scene}/rgb/'
    pathes = sorted([i for i in os.listdir(data_path) if i[0] == '0'])
    
    if isinstance(configs["train_idx"], str): # 0_K.png
        tmpl_idx_train = pathes.index(configs["train_idx"])    
        content_path = os.path.join(data_path, configs["train_idx"])
            
    if configs["train_idx"] == -1:
        configs["train_idx"] = np.random.randint(0, len(pathes))
    
    if configs["train_idx"] >= len(pathes):
        fail=True
    else:
        content_path = os.path.join(data_path, pathes[configs["train_idx"]])
    tmpl_idx_train = configs["train_idx"]
        
if fail:
    print("image invalid, please follow the instruction of the input config, thanks")
    
else:
    os.makedirs(os.path.join(configs['dst'], configs['style_name']), exist_ok=True)
    im = cv2.imread(content_path, 1)
    cv2.imwrite(os.path.join(configs['dst'], configs['style_name'], 'ref_content.png'), im)
    out_config = {
    "dataset_type": dataset,
    "scene_name": scene,
    "style_img": os.path.join(configs['dst'], configs['style_name'], 'style.png'),
    "tmpl_idx_train": tmpl_idx_train,
    "tmpl_idx_test": None,
    "color_pre": True,
    "style_name": configs['style_name']
    }
    with open(os.path.join(configs['dst'], configs['style_name'], 'data_config.json'), "w+") as f:
        f.write(json.dumps(out_config, indent=4))
    print(json.dumps(out_config, indent=4))