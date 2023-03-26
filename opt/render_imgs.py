# Copyright 2021 Alex Yu
# Eval

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from arguments import produce_render_args
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, viridis_no_norm_cmap
from util import config_util
from icecream import ic
import imageio
import cv2
from tqdm import tqdm
import json
import torch.nn.functional as F
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

if args.timing:
    args.no_vid = True
    args.ray_len = False

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')
want_metrics = True
if args.render_path:
    assert not args.train
    render_dir += '_path'
    want_metrics = False

# Handle various image transforms
if not args.render_path:
    # Do not crop if not render_path
    args.crop = 1.0
if args.crop != 1.0:
    render_dir += f'_crop{args.crop}'
if args.ray_len:
    render_dir += f'_raylen'
    want_metrics = False

dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)

if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_dir += '_nofg'

    # DEBUG
    #  grid.links.data[grid.links.size(0)//2:] = -1
    #  render_dir += "_chopx2"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

if not args.no_imsave:
    print('Will write out all frames as PNG (this take most of the time)')

im_style = torch.from_numpy(cv2.cvtColor(cv2.imread(args.style_img , 1), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0).to(device=device)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    # JULIAN: fit for NeRFDataset.
    if not hasattr(dset, "render_c2w"):
        dset.render_c2w = dset.c2w
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    n_images_gen = 0
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)

    frames = []
    #  im_gt_all = dset.gt.to(device=device)

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                           dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)
        im = grid.volume_render_image(cam, use_kernel=True, return_raylen=args.ray_len)
        if args.render_depth:
            maxval = 2.0
            minval = 0.4
            depth_img = (grid.volume_render_depth_image(cam, sigma_thresh=1e-8) - minval) / (maxval - minval)
            os.makedirs(path.join(render_dir, 'depth'), exist_ok=True)
            img_path = path.join(render_dir, 'depth', f'{img_id:04d}.png')
            depth_img = viridis_no_norm_cmap(depth_img.cpu().numpy())
            cv2.imwrite(img_path, (depth_img*255).astype(np.uint8))
        if args.render_xyz:
            rays = cam.gen_rays()
            all_positions = []
            for batch_start in range(0, cam.height * cam.width, 5000):
                depths = grid.volume_render_depth(rays[batch_start : batch_start + 5000], 1e-8)[..., None]
                # print(depths.shape, rays[batch_start : batch_start + 5000].dirs.shape)
                all_positions.append(rays[batch_start : batch_start + 5000].origins + rays[batch_start : batch_start + 5000].dirs*(depths.repeat(1, 3)))
            all_positions = torch.cat(all_positions, dim=0)
            # print(all_positions.shape)
            xyz = all_positions.view(cam.height, cam.width, 3)
            os.makedirs(path.join(render_dir, 'xyz'), exist_ok=True)
            img_path = path.join(render_dir, 'xyz', f'{img_id:04d}.npy')
            # print(img_path)
            np.save(img_path, xyz.cpu().numpy())
        if args.ray_len:
            minv, meanv, maxv = im.min().item(), im.mean().item(), im.max().item()
            im = viridis_cmap(im.cpu().numpy())
            cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                        0, 0.5, [255, 0, 0])
            im = torch.from_numpy(im).to(device=device)
        im.clamp_(0.0, 1.0)

        if not args.render_path:
            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            avg_psnr += psnr
            if not args.timing:
                ssim = compute_ssim(im_gt, im).item()
                avg_ssim += ssim
                print(img_id, 'PSNR', psnr, 'SSIM', ssim)
        img_path = path.join(render_dir, f'{img_id:04d}.png');
        im = im.cpu().numpy()
        if not args.render_path:
            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)
        if not args.timing:
            im = (im * 255).astype(np.uint8)
            if not args.no_imsave:
                imageio.imwrite(img_path,im)
            if not args.no_vid:
                frames.append(im)
        im = None
        n_images_gen += 1
    if want_metrics:
        print('AVERAGES')

        avg_psnr /= n_images_gen
        with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
            f.write(str(avg_psnr))
        print('PSNR:', avg_psnr)
        if not args.timing:
            avg_ssim /= n_images_gen
            print('SSIM:', avg_ssim)
            with open(path.join(render_dir, 'ssim.txt'), 'w') as f:
                f.write(str(avg_ssim))
    if not args.no_vid and len(frames):
        vid_path = render_dir + '.mp4'
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg


