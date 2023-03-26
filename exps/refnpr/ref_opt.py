# Copyright 2021 Alex Yu
# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:       sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>

import torch
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets, NeRFDataset, LLFFDataset, NSVFDataset, CO3DDataset
from util.util import get_expon_lr_func
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from arguments import produce_opt_args
from tqdm import tqdm
import cv2

from icecream import ic

# from style_transfer_losses import StyleTransferLosses
from ref_loss import NNFMLoss, match_colors_for_image_set


device = "cuda" if torch.cuda.is_available() else "cpu"

args = produce_opt_args()
data_dir = args.data_dir
with open(os.path.join(data_dir, "data_config.json")) as fp:
    style_dict = json.load(fp)

ic("Optimization step")
args.color_pre      = style_dict["color_pre"]
args.dataset_name   = style_dict["dataset_type"]
args.scene_name     = style_dict["scene_name"]
args.style_img      = style_dict["style_img"]
args.style_name     = style_dict["style_name"]
args.tmpl_idx_test  = style_dict["tmpl_idx_test"]
args.tmpl_idx_train = style_dict["tmpl_idx_train"]
args.data_dir       = f"./data/{args.dataset_name}/{args.scene_name}"
args.train_dir      = os.path.join(args.train_dir, args.style_name, "exp_out")
args.init_ckpt      = f"./exps/base_pr/ckpt_svox2/{args.dataset_name}/{args.scene_name}/ckpt.npz"
# args.init_ckpt      = f"./opt/ckpt_svox2/{args.dataset_type}/{args.scene_name}/ckpt.npz"
args.vgg_blocks     = [int(i) for i in str(args.vgg_blocks).split(",")]
args.exchange_tmp   = True
args.fast           = False

if args.tmpl_idx_train is not None:
    if not isinstance(args.tmpl_idx_train, list):
        args.tmpl_idx_train = [args.tmpl_idx_train]
elif not isinstance(args.tmpl_idx_test, list):
    args.tmpl_idx_test = [args.tmpl_idx_test]

ic(style_dict)
ic(args.vgg_blocks)
args.config_name = args.dataset_name
if "syn" in args.config_name:
     args.config_name = "syn"
    
args.config = f"./opt/configs/{args.config_name}_fixgeom.json"
if "no_fix" in args.train_dir:
    args.config = f"./opt/configs/{args.config_name}_flexible.json"
    
os.makedirs(args.train_dir, exist_ok=True)

config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, "args.json"), "w") as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, "opt_frozen.py"))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
    args.data_dir,
    split="train",
    device=device,
    factor=factor,
    n_images=args.n_train,
    **config_util.build_data_options(args),
)
if isinstance(dset, NSVFDataset):
    args.exchange_tmp  = True
    
assert dset.rays.origins.shape == (dset.n_images * dset.h * dset.w, 3)
assert dset.rays.dirs.shape == (dset.n_images * dset.h * dset.w, 3)

if args.background_nlayers > 0 and not dset.should_use_background:
    warn("Using a background model for dataset type " + str(type(dset)) + " which typically does not use background")

assert os.path.isfile(args.init_ckpt), "must specify a initial checkpoint"
grid = svox2.SparseGrid.load(args.init_ckpt, device=device, reset_basis_dim=args.reset_basis_dim)
ic("Loaded ckpt: ", args.init_ckpt)
ic(grid.basis_dim)

optim_basis_mlp = None

if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
    grid.reinit_learned_bases(init_type="sh")

elif grid.basis_type == svox2.BASIS_TYPE_MLP:
    # MLP!
    optim_basis_mlp = torch.optim.Adam(grid.basis_mlp.parameters(), lr=args.lr_basis)

grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print("Render options", grid.opt)

gstep_id_base = 0

lr_sigma_func = get_expon_lr_func(
    args.lr_sigma,
    args.lr_sigma_final,
    args.lr_sigma_delay_steps,
    args.lr_sigma_delay_mult,
    args.lr_sigma_decay_steps,
)
lr_sh_func = get_expon_lr_func(
    args.lr_sh,
    args.lr_sh_final,
    args.lr_sh_delay_steps,
    args.lr_sh_delay_mult,
    args.lr_sh_decay_steps,
)
lr_basis_func = get_expon_lr_func(
    args.lr_basis,
    args.lr_basis_final,
    args.lr_basis_delay_steps,
    args.lr_basis_delay_mult,
    args.lr_basis_decay_steps,
)
lr_sigma_bg_func = get_expon_lr_func(
    args.lr_sigma_bg,
    args.lr_sigma_bg_final,
    args.lr_sigma_bg_delay_steps,
    args.lr_sigma_bg_delay_mult,
    args.lr_sigma_bg_decay_steps,
)
lr_color_bg_func = get_expon_lr_func(
    args.lr_color_bg,
    args.lr_color_bg_final,
    args.lr_color_bg_delay_steps,
    args.lr_color_bg_delay_mult,
    args.lr_color_bg_decay_steps,
)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

last_upsamp_step = args.init_iters

if args.enable_random:
    warn("Randomness is enabled for training (normal for LLFF & scenes with background)")


###### resize style image such that its long side matches the long side of content images
if isinstance(args.style_img, str):
    args.style_img = [args.style_img]
style_imgs = [cv2.cvtColor(cv2.imread(args.style_img[i] , 1), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0 for i in range(len(args.style_img))]

for i in range(len(style_imgs)):
    style_imgs[i] = cv2.resize(style_imgs[i], (dset.w, dset.h), interpolation=cv2.INTER_CUBIC)

style_h, style_w = style_imgs[0].shape[:2]
content_long_side = max([dset.w, dset.h])

for i in range(len(style_imgs)):
    imageio.imwrite(
        os.path.join(args.train_dir, f"style_image_{i}.png"),
        np.clip(style_imgs[i] * 255.0, 0.0, 255.0).astype(np.uint8),
    )

if args.fast:
    style_imgs_d = [cv2.resize(m, (style_w//2, style_h//2), cv2.INTER_AREA) for m in style_imgs]
    style_imgs_o = [torch.from_numpy(m).to(device=device) for m in style_imgs]
    style_imgs = [torch.from_numpy(m).to(device=device) for m in style_imgs_d]
else:
    style_imgs_o = [torch.from_numpy(m).to(device=device) for m in style_imgs]
    style_imgs = style_imgs_o

global_start_time = datetime.now()

if not args.no_pre_ct:
    dset.rays.gt, color_tf = match_colors_for_image_set(dset.rays.gt, style_imgs[0])
    grid.apply_ct(color_tf.detach().cpu().numpy())

epoch_id = 0
epoch_size = None
batches_per_epoch = None
batch_size = None

nnfm_loss_fn = NNFMLoss(device=device)

if args.color_pre:
    # no color palette here, we apply a simple deeper matching scheme.
    loss_names=["tcm_loss", "color_patch", "content_loss"]
    args.vgg_blocks = [2,3,4]
else:
    loss_names=["tcm_loss"]
    args.vgg_blocks = [2,3,4]

# load template features.
tmpl_cams = []

if args.tmpl_idx_test is not None:
    for tmpl_id in args.tmpl_idx_test:
        dset_val = datasets[args.dataset_type](
            args.data_dir,
            split="test",
            device=device,
            factor=factor,
            n_images=args.n_train,
            **config_util.build_data_options(args),
        )
        dset_h, dset_w = dset_val.get_image_size(tmpl_id)
        # c2ws = dset_val.render_c2w.to(device=device)
        tmpl_cam = svox2.Camera(
            dset_val.render_c2w[tmpl_id].to(device=device),
            dset_val.intrins.get("fx", tmpl_id),
            dset_val.intrins.get("fy", tmpl_id),
            dset_val.intrins.get("cx", tmpl_id),
            dset_val.intrins.get("cy", tmpl_id),
            width=dset_w,
            height=dset_h,
            ndc_coeffs=dset_val.ndc_coeffs,
        )
        tmpl_cams += [tmpl_cam]
elif args.tmpl_idx_train is not None:
    for tmpl_id in args.tmpl_idx_train:
        dset_h, dset_w = dset.get_image_size(tmpl_id)
        tmpl_cam = svox2.Camera(
            dset.c2w[tmpl_id].to(device=device),
            dset.intrins.get('fx', tmpl_id),
            dset.intrins.get('fy', tmpl_id),
            dset.intrins.get('cx', tmpl_id),
            dset.intrins.get('cy', tmpl_id),
            dset_w, dset_h,
            ndc_coeffs=dset.ndc_coeffs)
        tmpl_cams += [tmpl_cam]
else:
    ic("You should specify a idx, program exit")
    exit()

# Use the depth-based multi-view color loss.
import copy
related_rays = copy.copy(dset.rays)
related_rays_gt = torch.load(f"{args.train_dir}/../color_corr.pt").reshape(-1)
related_rays.origins = dset.rays.origins.reshape(-1)[related_rays_gt!=-1].reshape(-1, 3).cuda()
related_rays.dirs = dset.rays.dirs.reshape(-1)[related_rays_gt!=-1].reshape(-1, 3).cuda()
related_rays_gt = related_rays_gt[related_rays_gt!=-1].reshape(-1, 3)
related_rays.gt = related_rays_gt.cuda()
related_rays_svox = svox2.Rays(related_rays.origins, related_rays.dirs)
ic(related_rays.origins.shape, related_rays.gt.shape, related_rays.dirs.shape)


while True:
    def train_step(optim_type):
        ic("Training epoch: ", epoch_id, epoch_size, batches_per_epoch, batch_size, optim_type)
        pbar = tqdm(enumerate(range(0, epoch_size, batch_size)), total=batches_per_epoch)
        for iter_id, batch_begin in pbar:
            stats = {}

            gstep_id = iter_id + gstep_id_base
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor

            if optim_type == "ray":
                """low frequency transfer"""
                batch_end = min(batch_begin + args.batch_size, epoch_size)
                batch_origins = dset.rays.origins[batch_begin:batch_end].to(device)
                batch_dirs = dset.rays.dirs[batch_begin:batch_end].to(device)
                rgb_gt = dset.rays.gt[batch_begin:batch_end].to(device)
                rays = svox2.Rays(batch_origins, batch_dirs)

                rgb_pred = grid.volume_render_fused(
                    rays,
                    rgb_gt,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random,
                    is_rgb_gt=True,
                    reset_grad_indexers=True,
                )
                # add a eps to avoid div by zero if no pre-colorizations.
                eps = 1e-8
                mse = F.mse_loss(rgb_gt, rgb_pred) + eps
                psnr = -10.0 * math.log10(mse.detach().item())
                stats["psnr"] = psnr
            elif optim_type == "image":
                num_views, view_height, view_width = dset.n_images, dset.h, dset.w
                img_id = np.random.randint(low=0, high=num_views)
                rays = svox2.Rays(
                    dset.rays.origins.view(num_views, view_height * view_width, 3)[img_id].to(device),
                    dset.rays.dirs.view(num_views, view_height * view_width, 3)[img_id].to(device),
                )

                # Add a template color loss.
                for i in range(len(style_imgs_o)):
                    tmpl_gt_sample = style_imgs_o[i].view(-1, 3).contiguous()
                    idx = np.random.randint(len(tmpl_gt_sample), size=args.n_samples//len(style_imgs_o))
                    tmpl_gt_sample = tmpl_gt_sample[idx]
                    rays_tmpl = tmpl_cams[i].gen_rays()[idx]
                    tmpl_sample_pred = grid.volume_render_fused(
                        rays_tmpl,
                        tmpl_gt_sample,
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random,
                        is_rgb_gt=True,
                        reset_grad_indexers=True,
                    )

                # Use the depth-based multi-view color loss.
                idx = np.random.randint(len(related_rays.origins), size=args.n_samples)
                rays_tmpl = related_rays_svox[idx]
                tmpl_gt_sample = related_rays.gt[idx]
                tmpl_sample_pred = grid.volume_render_fused(
                    rays_tmpl,
                    tmpl_gt_sample,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random,
                    is_rgb_gt=True,
                    reset_grad_indexers=True,
                )

                def compute_image_loss():
                    with torch.no_grad():
                        cam = svox2.Camera(
                            dset.c2w[img_id].to(device=device),
                            dset.intrins.get("fx", img_id),
                            dset.intrins.get("fy", img_id),
                            dset.intrins.get("cx", img_id),
                            dset.intrins.get("cy", img_id),
                            width=view_width,
                            height=view_height,
                            ndc_coeffs=dset.ndc_coeffs,
                        )
                        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
                        rgb_gt = dset.rays.gt.view(num_views, view_height, view_width, 3)[img_id].to(
                            device
                        )
                        rgb_gt = rgb_gt.permute(2, 0, 1).unsqueeze(0).contiguous()
                        rgb_pred = rgb_pred.permute(2, 0, 1).unsqueeze(0).contiguous()
                    
                    rgb_pred.requires_grad_(True)
                    w_variance = torch.mean(torch.pow(rgb_pred[:, :, :, :-1] - rgb_pred[:, :, :, 1:], 2))
                    h_variance = torch.mean(torch.pow(rgb_pred[:, :, :-1, :] - rgb_pred[:, :, 1:, :], 2))
                    img_tv_loss = args.img_tv_weight * (h_variance + w_variance) / 2.0
                    if args.fast:
                        loss_dict = nnfm_loss_fn(
                            outputs=F.interpolate(
                                rgb_pred,
                                size=None,
                                scale_factor=0.5,
                                mode="bilinear",
                            ),
                            styles=None,
                            blocks=args.vgg_blocks,
                            loss_names=loss_names,
                            contents=F.interpolate(
                                rgb_gt,
                                size=None,
                                scale_factor=0.5,
                                mode="bilinear",
                            ),
                        )
                    else:
                        loss_dict = nnfm_loss_fn(
                            outputs=rgb_pred,
                            styles=None,
                            blocks=args.vgg_blocks,
                            loss_names=loss_names,
                            contents=rgb_gt,
                        )
                    if "content_loss" in loss_dict:
                        loss_dict["content_loss"] *= args.content_weight
                    loss_dict["img_tv_loss"] = img_tv_loss
                    
                    loss = sum(list(loss_dict.values()))
                    loss.backward()
                    rgb_pred_grad = rgb_pred.grad.squeeze(0).permute(1, 2, 0).contiguous().clone().detach().view(-1, 3)
                    
                    return rgb_pred_grad, loss_dict

                rgb_pred_grad, loss_dict = compute_image_loss()
                rgb_pred = []
                grid.alloc_grad_indexers()
                for view_batch_start in range(0, view_height * view_width, args.batch_size):
                    rgb_pred_patch = grid.volume_render_fused(
                        rays[view_batch_start : view_batch_start + args.batch_size],
                        rgb_pred_grad[view_batch_start : view_batch_start + args.batch_size],
                        beta_loss=args.lambda_beta,
                        sparsity_loss=args.lambda_sparsity,
                        randomize=args.enable_random,
                        is_rgb_gt=False,
                        reset_grad_indexers=False,
                    )
                    rgb_pred.append(rgb_pred_patch.clone().detach())
                rgb_pred = torch.cat(rgb_pred, dim=0).reshape(view_height, view_width, 3)
                for x in loss_dict:
                    stats[x] = loss_dict[x].item()

            if (iter_id + 1) % args.print_every == 0:
                log_str = ""
                for stat_name in stats:
                    summary_writer.add_scalar(stat_name, stats[stat_name], global_step=gstep_id)
                    log_str += "{:.4f} ".format(stats[stat_name])
                pbar.set_description(f"{gstep_id} {log_str}")

                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                    summary_writer.add_scalar("lr_basis", lr_basis, global_step=gstep_id)
                if grid.use_background:
                    summary_writer.add_scalar("lr_sigma_bg", lr_sigma_bg, global_step=gstep_id)
                    summary_writer.add_scalar("lr_color_bg", lr_color_bg, global_step=gstep_id)

            if args.weight_decay_sh < 1.0:
                grid.sh_data.data *= args.weight_decay_sigma
            if args.weight_decay_sigma < 1.0:
                grid.density_data.data *= args.weight_decay_sh

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                grid.inplace_tv_grad(
                    grid.density_data.grad,
                    scaling=args.lambda_tv,
                    sparse_frac=args.tv_sparsity,
                    logalpha=args.tv_logalpha,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_sh > 0.0:
                grid.inplace_tv_color_grad(
                    grid.sh_data.grad,
                    scaling=args.lambda_tv_sh,
                    sparse_frac=args.tv_sh_sparsity,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_lumisphere > 0.0:
                grid.inplace_tv_lumisphere_grad(
                    grid.sh_data.grad,
                    scaling=args.lambda_tv_lumisphere,
                    dir_factor=args.tv_lumisphere_dir_factor,
                    sparse_frac=args.tv_lumisphere_sparsity,
                    ndc_coeffs=dset.ndc_coeffs,
                )
            if args.lambda_l2_sh > 0.0:
                grid.inplace_l2_color_grad(grid.sh_data.grad, scaling=args.lambda_l2_sh)
            if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                grid.inplace_tv_background_grad(
                    grid.background_data.grad,
                    scaling=args.lambda_tv_background_color,
                    scaling_density=args.lambda_tv_background_sigma,
                    sparse_frac=args.tv_background_sparsity,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_basis > 0.0:
                tv_basis = grid.tv_basis()
                loss_tv_basis = tv_basis * args.lambda_tv_basis
                loss_tv_basis.backward()

            # Manual SGD/rmsprop step
            # ic(lr_sigma)
            if lr_sigma > 0.0:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
            grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)

            if grid.use_background:
                grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)

            if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
            elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                optim_basis_mlp.step()
                optim_basis_mlp.zero_grad()

    img_id = np.random.randint(low=0, high=dset.n_images)
    cam = svox2.Camera(
        dset.c2w[img_id].to(device=device),
        dset.intrins.get("fx", img_id),
        dset.intrins.get("fy", img_id),
        dset.intrins.get("cx", img_id),
        dset.intrins.get("cy", img_id),
        width=dset.get_image_size(img_id)[1],
        height=dset.get_image_size(img_id)[0],
        ndc_coeffs=dset.ndc_coeffs,
    )
    rgb_pred = grid.volume_render_image(cam, use_kernel=True).detach().cpu().numpy()
    imageio.imwrite(
        os.path.join(args.train_dir, f"logim_{epoch_id}.png"),
        np.clip(rgb_pred * 255.0, 0.0, 255.0).astype(np.uint8),
    )
        
    if epoch_id < args.mse_num_epoches:
        epoch_size = dset.rays.origins.size(0)
        batch_size = args.batch_size
        batches_per_epoch = (epoch_size - 1) // batch_size + 1
        train_step(optim_type="ray")

    if epoch_id == args.mse_num_epoches + int(args.nnfm_num_epoches*0.75):
        loss_names = ['online_tmp_loss']
        args.exchange_tmp = True

    # Now, no need for using the original domain to optimize the second stage.
    # TODO      | If we still need a pre-colorization, work on it.
    if (epoch_id == args.mse_num_epoches or epoch_id == args.mse_num_epoches + int(args.nnfm_num_epoches*0.75)) and\
        args.exchange_tmp:
        num_views, view_height, view_width = dset.n_images, dset.h, dset.w
        rgb_gt = dset.rays.gt.view(num_views, view_height, view_width, 3)
        bs = view_height*view_width
        with torch.no_grad():
            for img_id in range(num_views):
                cam = svox2.Camera(
                    dset.c2w[img_id].to(device=device),
                    dset.intrins.get("fx", img_id),
                    dset.intrins.get("fy", img_id),
                    dset.intrins.get("cx", img_id),
                    dset.intrins.get("cy", img_id),
                    width=view_width,
                    height=view_height,
                    ndc_coeffs=dset.ndc_coeffs,
                )
                rgb_pred = grid.volume_render_image(cam, use_kernel=True)
                dset.rays.gt[bs*img_id: bs*img_id + bs] = rgb_pred.view(-1, 3)
        tmpl_imgs = []
        for tmpl_cam in tmpl_cams:
            tmpl_imgs += [grid.volume_render_image(tmpl_cam, use_kernel=True).permute(2,0,1).unsqueeze(0)]
        if args.fast:
            tmpl_imgs = [F.interpolate(tmpl, size=None, scale_factor=0.5, mode="bilinear") for tmpl in tmpl_imgs]
        nnfm_loss_fn.preload_golden_template(tmpl_imgs, 
                                     style_imgs,
                                     blocks=args.vgg_blocks)
        args.exchange_tmp = False
      
    if args.mse_num_epoches <= epoch_id and epoch_id < args.mse_num_epoches + args.nnfm_num_epoches:
        epoch_size = dset.n_images
        batch_size = 1
        batches_per_epoch = (dset.n_images - 1) // batch_size + 1
        train_step(optim_type="image")

    epoch_id += 1
    gstep_id_base += batches_per_epoch
    torch.cuda.empty_cache()
    gc.collect()

    if epoch_id >= args.mse_num_epoches + args.nnfm_num_epoches:

        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, "time_mins.txt"), "w")
        timings_file.write(f"{secs / 60}\n")
        timings_file.close()

        ckpt_path = path.join(args.train_dir, "ckpt.npz")
        grid.save(ckpt_path)

        img_id = np.random.randint(low=0, high=dset.n_images)
        cam = svox2.Camera(
            dset.c2w[img_id].to(device=device),
            dset.intrins.get("fx", img_id),
            dset.intrins.get("fy", img_id),
            dset.intrins.get("cx", img_id),
            dset.intrins.get("cy", img_id),
            width=dset.get_image_size(img_id)[1],
            height=dset.get_image_size(img_id)[0],
            ndc_coeffs=dset.ndc_coeffs,
        )
        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
        rgb_pred = rgb_pred.detach().cpu().numpy()

        imageio.imwrite(
            os.path.join(args.train_dir, f"logim_{epoch_id}_final.png"),
            np.clip(rgb_pred * 255.0, 0.0, 255.0).astype(np.uint8),
        )

        break
