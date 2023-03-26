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
from exps.arguments import produce_opt_args
from tqdm import tqdm
import cv2

from icecream import ic

# from style_transfer_losses import StyleTransferLosses
from exps.ref_loss import NNFMLoss, match_colors_for_image_set


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

style_img = imageio.imread(args.style_img[0], pilmode="RGB").astype(np.float32) / 255.0
style_h, style_w = style_img.shape[:2]
style_img = cv2.resize(
    style_img,
    (style_img.shape[1] // 2, style_img.shape[0] // 2),
    interpolation=cv2.INTER_AREA,
)
imageio.imwrite(
    os.path.join(args.train_dir, "style_image.png"),
    np.clip(style_img * 255.0, 0.0, 255.0).astype(np.uint8),
)

style_img = torch.from_numpy(style_img).to(device=device)
ic("Style image: ", args.style_img[0], style_img.shape)

global_start_time = datetime.now()

if not args.no_pre_ct:
    dset.rays.gt, color_tf = match_colors_for_image_set(dset.rays.gt, style_img)
    grid.apply_ct(color_tf.detach().cpu().numpy())

epoch_id = 0
epoch_size = None
batches_per_epoch = None
batch_size = None

nnfm_loss_fn = NNFMLoss(device=device)

loss_names=["nnfm_loss", "content_loss"]
args.vgg_blocks = [2]

while True:
    def train_step(optim_type):
        ic("Training epoch: ", epoch_id, "Stylizing training images first")
        stylized_imgs = []
        img_id = 0
        if optim_type != "ray":
            num_views, view_height, view_width = dset.n_images, dset.h, dset.w
            rgb_gts = dset.rays.gt.view(num_views, view_height, view_width, 3)
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
                    rgb_gts[img_id] = rgb_pred.reshape(view_height, view_width, 3).contiguous().cpu().clamp_(0.0, 1.0)

            for rgb_gt in rgb_gts:
                stylized_gt = rgb_gt.clone().unsqueeze(0).permute(0,3,1,2).contiguous().to(device=device)
                stylized_gt = torch.nn.Parameter(stylized_gt, requires_grad=True)
                style_img_optimizer = torch.optim.LBFGS([stylized_gt])
                
                # iteratively update the stylization training set.
                for _ in range(10):
                    def closure():
                        style_img_optimizer.zero_grad()
                        loss_dict = nnfm_loss_fn(
                            F.interpolate(
                                stylized_gt,
                                size=None,
                                scale_factor=0.5,
                                mode="bilinear",
                            ),
                            style_img.permute(2, 0, 1).unsqueeze(0),
                            blocks=[
                                1,2,3,4
                            ],
                            loss_names=["gram_loss", "content_loss"],
                            contents=F.interpolate(
                                rgb_gt.unsqueeze(0).permute(0,3,1,2).to(device=device),
                                size=None,
                                scale_factor=0.5,
                                mode="bilinear",
                            ),
                        )
                        loss_dict["content_loss"] *= args.content_weight
                        loss = loss_dict["gram_loss"] + loss_dict["content_loss"]
                        loss.backward(retain_graph=True)
                        return loss
                    style_img_optimizer.step(closure)
                os.makedirs(os.path.join(args.train_dir, f"stylized_res_{epoch_id}"), exist_ok = True)
                
                imageio.imwrite(
                    os.path.join(args.train_dir, f"stylized_res_{epoch_id}", f"data_{img_id}.png"),
                    np.clip(stylized_gt[0].permute(1,2,0).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
                )
                img_id += 1
                stylized_imgs.append(stylized_gt[0].permute(1,2,0))
            dset.stylized_gt_update(stylized_imgs)

        ic("Training epoch: ", epoch_id, "Then finetune a nerf.")
        pbar = tqdm(enumerate(range(0, epoch_size, batch_size)), total=batches_per_epoch)
        
        for iter_id, batch_begin in pbar:
            stats = {}
            gstep_id = iter_id + gstep_id_base
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            # ic(lr_sh, lr_sigma)
            lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
            lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
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
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(
                    grid.density_data.grad,
                    scaling=args.lambda_tv,
                    sparse_frac=args.tv_sparsity,
                    logalpha=args.tv_logalpha,
                    ndc_coeffs=dset.ndc_coeffs,
                    contiguous=args.tv_contiguous,
                )
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
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
      
    if args.mse_num_epoches <= epoch_id and epoch_id < args.mse_num_epoches + args.nnfm_num_epoches:
        epoch_size = dset.rays.origins.size(0)
        batch_size = args.batch_size
        batches_per_epoch = (epoch_size - 1) // batch_size + 1
        train_step(optim_type="image")

    epoch_id += 1
    gstep_id_base += batches_per_epoch
    torch.cuda.empty_cache()
    gc.collect()

    if epoch_id >= args.mse_num_epoches + args.nnfm_num_epoches:    
        if not args.no_post_ct:
            num_views, view_height, view_width = dset.n_images, dset.h, dset.w
            rgb_gt = dset.rays.gt.view(num_views, view_height, view_width, 3)
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
                    rgb_gt[img_id] = rgb_pred.reshape(view_height, view_width, 3).contiguous().cpu().clamp_(0.0, 1.0)
            dset.rays.gt, color_tf = match_colors_for_image_set(dset.rays.gt, style_img)
            grid.apply_ct(color_tf.detach().cpu().numpy())

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
