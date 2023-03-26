# a function for train inference linking (R^3).
from math import floor
import cv2
import imageio
import numpy as np
from svox2 import Rays
import torch
import svox2
import os
import json
from tqdm import tqdm
# from opt.util.nerf_dataset import NeRFDataset

from util.dataset import datasets, NSVFDataset, NeRFDataset
from util import config_util
from exps.arguments import produce_args
from icecream import ic

def cos_dist(vec_a, vec_b):
    return torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0)

args = produce_args()
# post-assign args 
data_dir = args.data_dir

# load json file.
with open(os.path.join(data_dir, "data_config.json")) as fp:
    style_dict = json.load(fp)
    
args.color_pre      = style_dict["color_pre"]
args.dataset_type   = style_dict["dataset_type"]
args.scene_name     = style_dict["scene_name"]
args.style_img      = style_dict["style_img"]
args.style_name     = style_dict["style_name"]
args.tmpl_idx_test  = style_dict["tmpl_idx_test"]
args.tmpl_idx_train = style_dict["tmpl_idx_train"]
args.data_dir       = "./data/{}/{}".format(args.dataset_type, args.scene_name)
args.out_dir        = os.path.join(args.out_dir, args.style_name)

if args.tmpl_idx_train is not None:
    if not isinstance(args.tmpl_idx_train, list):
        args.tmpl_idx_train = [args.tmpl_idx_train]
elif not isinstance(args.tmpl_idx_test, list):
    args.tmpl_idx_test = [args.tmpl_idx_test]

os.makedirs(args.out_dir, exist_ok=True)

ic("Start ray registeration step")
if os.path.exists(os.path.join(args.out_dir, "color_corr.pt")):
    ic("preload the correspondence, goto the next step")
    PRELOAD=True
    exit()
ic(style_dict)

pretrained_svox2 = "./exps/base_pr/ckpt_svox2/{}/{}/ckpt.npz".format(args.dataset_type, args.scene_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

# for multiple reference.
if isinstance(args.style_img, str):
    args.style_img = [args.style_img]
style_imgs = [cv2.cvtColor(cv2.imread(args.style_img[i] , 1), cv2.COLOR_BGR2RGB)/255.0 for i in range(len(args.style_img))]


grid = svox2.SparseGrid.load(pretrained_svox2, device=device).eval()
ic("Loaded ckpt: ", pretrained_svox2)
dset = datasets['auto'](args.data_dir, split="train",
                        **config_util.build_data_options(args))

for i in range(len(style_imgs)):
    style_imgs[i] = cv2.resize(style_imgs[i], (dset.w, dset.h), interpolation=cv2.INTER_CUBIC)

# two options: for llff we can use the validation view for an easier index match.
tmpl_cams = []
if args.tmpl_idx_train is not None:
    with torch.no_grad():
        for idx_train in args.tmpl_idx_train:
            tmpl_cam = svox2.Camera(dset.c2w[idx_train].to(device=device),
                            dset.intrins.get('fx', idx_train),
                            dset.intrins.get('fy', idx_train),
                            dset.intrins.get('cx', idx_train),
                            dset.intrins.get('cy', idx_train),
                            dset.w, dset.h,
                            ndc_coeffs=dset.ndc_coeffs)
            im = grid.volume_render_image(tmpl_cam, use_kernel=True, return_raylen=args.ray_len)
            tmpl_cams += [tmpl_cam]

    tmpl_ids = args.tmpl_idx_train
    
elif args.tmpl_idx_test is not None:
    dset_val = datasets['auto'](
        args.data_dir,
        split="test",
        device=device,
        **config_util.build_data_options(args),
    )
    with torch.no_grad():
        for idx_test in args.tmpl_idx_test:
            tmpl_cam = svox2.Camera(
                dset_val.render_c2w[idx_test].to(device=device),
                dset_val.intrins.get("fx", idx_test),
                dset_val.intrins.get("fy", idx_test),
                dset_val.intrins.get("cx", idx_test),
                dset_val.intrins.get("cy", idx_test),
                dset.w, dset.h,
                ndc_coeffs=dset_val.ndc_coeffs,
            )
            tmpl_cams += [tmpl_cam]
            im = grid.volume_render_image(tmpl_cam, use_kernel=True, return_raylen=args.ray_len)

    tmpl_ids = args.tmpl_idx_test
    
else:
    ic("You should specify a idx, program exit")
    exit()

# Here, what we want to do is establish a fine lookup table between
# the training data and the stylized templates. This can be divided into the following three steps:

# Step A: First, pre-store all the information we need: depth, position, and position range.
eps = 1e-5

sigma_thresh = 1e-8
if isinstance(dset, NSVFDataset):
    sigma_thresh = 100
ic(sigma_thresh)
    
depths = []
xyzs = []
xyz_max = torch.zeros(3).cuda()
xyz_min = torch.ones(3).cuda()

for n in tqdm(range(dset.n_images)):
    cam = svox2.Camera(dset.c2w[n].to(device=device),
                    dset.intrins.get('fx', n),
                    dset.intrins.get('fy', n),
                    dset.intrins.get('cx', n),
                    dset.intrins.get('cy', n),
                    dset.w, dset.h,
                    ndc_coeffs=dset.ndc_coeffs)
    rays = cam.gen_rays()
    depth_img = grid.volume_render_depth_image(cam, sigma_thresh=sigma_thresh)[..., None]
    depth_rep = depth_img.repeat(1, 1, 3)
    xyz_pos = rays.origins.view(cam.height, cam.width, 3) + \
            rays.dirs.view(cam.height, cam.width, 3)*depth_rep
    xyz_min = torch.minimum(xyz_min, torch.min(xyz_pos[depth_rep!=0].reshape(-1, 3), dim=0)[0])
    xyz_max = torch.maximum(xyz_max, torch.max(xyz_pos[depth_rep!=0].reshape(-1, 3), dim=0)[0])
    xyz_pos[depth_rep==0] = 0
    depths += depth_img.unsqueeze(0)
    xyzs += xyz_pos.unsqueeze(0)

depths = torch.cat(depths).cpu().numpy().reshape(dset.n_images, dset.w, dset.h)
xyzs = torch.cat(xyzs).cpu().numpy().reshape(dset.n_images, dset.w, dset.h, -1)
aabb = torch.cat((xyz_min, xyz_max)).cpu().numpy()
ic(aabb)
    
# Step B. For each point, we can find its corresponding geometric position, 
# and store template information 
# (xyz + direction, somewhat similar to the approach used in visualization tools) in a 3D grid (256^3).
n_grid = 512
grid_unit = 1.0 / n_grid
grid_offset = 1.0 / n_grid * 0.5
eps = 1e-5

xyz_min = aabb[:3]
xyz_max = aabb[3:]

position_dict = (torch.ones((n_grid, n_grid, n_grid, 3))*-2).cuda()
direction_dict = (torch.ones((n_grid, n_grid, n_grid, 3))*-2).cuda()
id_dict = (torch.ones((n_grid, n_grid, n_grid))*-2).cuda().long()
position_counts = torch.zeros((n_grid, n_grid, n_grid)).long()

xyzs = (xyzs - xyz_min) / (xyz_max - xyz_min + eps)
dirs = dset.rays.dirs.reshape(dset.n_images, -1, 3).cuda()
tmpl_xyzs = []
tmpl_depths = []
tmpl_dirs = []
tmpl_ignore = []
for i, tmpl_id in enumerate(tmpl_ids):
    if args.tmpl_idx_train is not None:
        tmpl_xyzs += [xyzs[tmpl_id].reshape(-1, 3)]
        depth_img = depths[tmpl_id].reshape(-1)
        tmpl_depths += [depth_img]
        tmpl_dirs += [dirs[tmpl_id]]
    else:
        # with the test dataset.
        rays = tmpl_cam.gen_rays()
        depth_img = grid.volume_render_depth_image(tmpl_cam, sigma_thresh=sigma_thresh)[..., None]
        depth_rep = depth_img.repeat(1, 1, 3)
        depth_img = depth_img.reshape(-1).cpu().numpy()
        tmpl_xyz = (rays.origins.view(tmpl_cam.height, tmpl_cam.width, 3) + \
                    rays.dirs.view(tmpl_cam.height, tmpl_cam.width, 3)*depth_rep).cpu().numpy()
        tmpl_xyz = (tmpl_xyz - xyz_min) / (xyz_max - xyz_min + eps)
        
        tmpl_dirs += (rays.dirs.reshape(-1, 3))
        tmpl_depths += [depth_img]
        tmpl_xyzs += [tmpl_xyz.reshape(-1, 3)]

    # trick：ignore boundary with a mask. 
    border_ignore_mask = depth_img.reshape(dset.h, dset.w)
    if isinstance(dset, NeRFDataset):
        border_ignore_mask[border_ignore_mask != 0] = 1
        border_ignore_mask = cv2.erode(border_ignore_mask, np.ones((3, 3)), iterations=7)
    tmpl_ignore += [border_ignore_mask.reshape(-1)]

tmpl_xyz = np.concatenate(tmpl_xyzs, axis=0)
tmpl_depth = np.concatenate(tmpl_depths, axis=0)
tmpl_dir = torch.concat(tmpl_dirs, dim=0)
if len(tmpl_dir.shape) == 1:
    tmpl_dir = tmpl_dir.view(-1, 3)
tmpl_ignore = np.concatenate(tmpl_ignore, axis=0)
ic(tmpl_xyz.shape, tmpl_depth.shape, tmpl_dir.shape, tmpl_ignore.shape)

preload = False 
tmpl_xyz = torch.from_numpy(tmpl_xyz).cuda()

# TODO: accelerate here. We use a denser voxels and deprecate the density. 512^3 = 256^3*8
ray_batch_size = 20000
h = dset.h
w = dset.w

for i in tqdm(range((len(tmpl_xyz))//ray_batch_size + (int)((len(tmpl_xyz))%ray_batch_size!=0))):

    xyz_batch = tmpl_xyz[i*ray_batch_size:i*ray_batch_size + ray_batch_size]
    ref_dir_batch = tmpl_dir[i*ray_batch_size:i*ray_batch_size + ray_batch_size]
    id_batch = torch.tensor([m for m in range(i*ray_batch_size, min((i+1)*ray_batch_size, len(tmpl_xyz)))]).long().cuda()
    index_int = (xyz_batch*n_grid).long()
    position_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]] = xyz_batch
    direction_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]] = ref_dir_batch
    id_dict[index_int[:, 0], index_int[:, 1], index_int[:, 2]] = id_batch
    
torch.cuda.empty_cache()

# Step C. Next, in the training rays, find the top K values (currently, k=3) 
# or index values using nearest neighbor search, and use them as the corresponding ray results 
# (Verification: A simple verification can be done using rendering and pixel replacement).

res_dict = (torch.ones((dset.n_images, w*h, 3))*-1).cuda()
for test_idx in tqdm(range(dset.n_images)):
    with torch.no_grad():
        cam_ref = svox2.Camera(dset.c2w[test_idx].to(device=device),
                            dset.intrins.get('fx', test_idx),
                            dset.intrins.get('fy', test_idx),
                            dset.intrins.get('cx', test_idx),
                            dset.intrins.get('cy', test_idx),
                            dset.w, dset.h,
                            ndc_coeffs=dset.ndc_coeffs)
        im = grid.volume_render_image(cam_ref, use_kernel=True, return_raylen=args.ray_len)
    rays = cam_ref.gen_rays()

    ref_xyz = torch.from_numpy(xyzs[test_idx].reshape(-1, 3)).cuda()
    ref_depth =  torch.from_numpy(depths[test_idx].reshape(-1)).cuda()
    ref_dir = dirs[test_idx]

    ref_img = dset.rays.gt.reshape(-1, h*w, 3)[test_idx].cuda()
    style_tmpl = torch.from_numpy(np.concatenate([m.reshape(-1, 3) for m in style_imgs], axis=0)).cuda()
    # ic(style_tmpl.shape)
    ray_batch_size = 20000
    
    for i in range((h*w)//ray_batch_size + (int)((h*w)%ray_batch_size!=0)):
        xyz_batch = ref_xyz[i*ray_batch_size:i*ray_batch_size + ray_batch_size]
        ref_dir_batch = ref_dir[i*ray_batch_size:i*ray_batch_size + ray_batch_size]

        index_int = (xyz_batch*n_grid).long()
        grid_pos = position_dict[index_int[:,0], index_int[:,1], index_int[:,2]]
        grid_dir = direction_dict[index_int[:,0], index_int[:,1], index_int[:,2]]
        grid_idx = id_dict[index_int[:,0], index_int[:,1], index_int[:,2]]
        cur_dist = torch.sum((xyz_batch - grid_pos)**2, dim=1)
        # mask with a direction distance restriction.
        dir_dist = torch.nn.functional.cosine_similarity(ref_dir_batch, grid_dir, dim=1)
        cur_dist[dir_dist < 0.3] = 100

        res_dict[test_idx][i*ray_batch_size:i*ray_batch_size + ray_batch_size] = style_tmpl[grid_idx]
        res_dict[test_idx][i*ray_batch_size:i*ray_batch_size + ray_batch_size][cur_dist > 0.1] = -1
    
# Save the dictionary.
# TODO: efficient data structure to save memory (save as PNG?)
torch.save(res_dict.cpu(), os.path.join(args.out_dir, 'color_corr.pt'))
# DONE