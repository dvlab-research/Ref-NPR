# Copyright 2021 Alex Yu
# Render 360 circle path

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from exps.arguments import produce_render_args
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, pose_spherical
from util import config_util
from icecream import ic

import imageio
import cv2
from tqdm import tqdm
import json

args = produce_render_args()
with open(os.path.join(args.data_dir, "data_config.json")) as fp:
    style_dict = json.load(fp)

args.color_pre      = style_dict["color_pre"]
args.dataset_name   = style_dict["dataset_type"]
args.scene_name     = style_dict["scene_name"]
args.style_img      = style_dict["style_img"]
args.style_name     = style_dict["style_name"]
args.data_dir       = f"./data/{args.dataset_name}/{args.scene_name}"
args.ckpt           = path.join(args.ckpt, args.style_name, 'exp_out', 'ckpt.npz')
ic(args.ckpt)

config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'


dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))

if args.vec_up is None:
    up_rot = dset.c2w[:, :3, :3].cpu().numpy()
    ups = np.matmul(up_rot, np.array([0, -1.0, 0])[None, :, None])[..., 0]
    args.vec_up = np.mean(ups, axis=0)
    args.vec_up /= np.linalg.norm(args.vec_up)
    print('  Auto vec_up', args.vec_up)
else:
    args.vec_up = np.array(list(map(float, args.vec_up.split(","))))


args.offset = np.array(list(map(float, args.offset.split(","))))
if args.traj_type == 'spiral':
    angles = np.linspace(-180, 180, args.num_views + 1)[:-1]
    elevations = np.linspace(args.elevation, args.elevation2, args.num_views)
    c2ws = [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(elevations, angles)
    ]
    c2ws += [
        pose_spherical(
            angle,
            ele,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for ele, angle in zip(reversed(elevations), angles)
    ]
else :
    c2ws = [
        pose_spherical(
            angle,
            args.elevation,
            args.radius,
            args.offset,
            vec_up=args.vec_up,
        )
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ]
c2ws = np.stack(c2ws, axis=0)
if args.vert_shift != 0.0:
    c2ws[:, :3, 3] += np.array(args.vec_up) * args.vert_shift
c2ws = torch.from_numpy(c2ws).to(device=device)

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_out_path = path.join(path.dirname(args.ckpt), 'circle_renders')

# Handle various image transforms
if args.crop != 1.0:
    render_out_path += f'_crop{args.crop}'
if args.vert_shift != 0.0:
    render_out_path += f'_vshift{args.vert_shift}'

grid = svox2.SparseGrid.load(args.ckpt, device=device)
print(grid.center, grid.radius)

# DEBUG
#  grid.background_data.data[:, 32:, -1] = 0.0
#  render_out_path += '_front'

if grid.use_background:
    if args.nobg:
        grid.background_data.data[..., -1] = 0.0
        render_out_path += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_out_path += '_nofg'

    #  # DEBUG
    #  grid.background_data.data[..., -1] = 100.0
    #  a1 = torch.linspace(0, 1, grid.background_data.size(0) // 2, dtype=torch.float32, device=device)[:, None]
    #  a2 = torch.linspace(1, 0, (grid.background_data.size(0) - 1) // 2 + 1, dtype=torch.float32, device=device)[:, None]
    #  a = torch.cat([a1, a2], dim=0)
    #  c = torch.stack([a, 1-a, torch.zeros_like(a)], dim=-1)
    #  grid.background_data.data[..., :-1] = c
    #  render_out_path += "_gradient"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_out_path += '_blackbg'
    grid.opt.background_brightness = 0.0

render_out_path += '.mp4'
print('Writing to', render_out_path)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = c2ws.size(0)
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    frames = []
    #  if args.near_clip >= 0.0:
    grid.opt.near_clip = 0.0 #args.near_clip
    if args.width is None:
        args.width = dset.get_image_size(0)[1]
    if args.height is None:
        args.height = dset.get_image_size(0)[0]

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = args.height, args.width
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', 0),
                           dset.intrins.get('fy', 0),
                           w * 0.5,
                           h * 0.5,
                           w, h,
                           ndc_coeffs=(-1.0, -1.0))
        torch.cuda.synchronize()
        im = grid.volume_render_image(cam, use_kernel=True)
        torch.cuda.synchronize()
        im.clamp_(0.0, 1.0)

        im = im.cpu().numpy()
        im = (im * 255).astype(np.uint8)
        frames.append(im)
        im = None
        n_images_gen += 1
    if len(frames):
        vid_path = render_out_path
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg


