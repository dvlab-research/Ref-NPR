from util import config_util
import argparse

def produce_render_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)

    config_util.define_common_args(parser)

    parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
    parser.add_argument('--train', action='store_true', default=False, help='render train set')
    parser.add_argument('--render_path',
                        action='store_true',
                        default=False,
                        help="Render path instead of test images (no metrics will be given)")
    parser.add_argument('--timing',
                        action='store_true',
                        default=False,
                        help="Run only for timing (do not save images or use LPIPS/SSIM; "
                        "still computes PSNR to make sure images are being generated)")
    parser.add_argument('--no_lpips',
                        action='store_true',
                        default=False,
                        help="Disable LPIPS (faster load)")
    parser.add_argument('--no_vid',
                        action='store_true',
                        default=False,
                        help="Disable video generation")
    parser.add_argument('--no_imsave',
                        action='store_true',
                        default=False,
                        help="Disable image saving (can still save video; MUCH faster)")
    parser.add_argument('--fps',
                        type=int,
                        default=30,
                        help="FPS of video")
    parser.add_argument('--render_depth',
                        action='store_true',
                        default=False,
                        help="whether render depth image")
    parser.add_argument('--render_xyz',
                        action='store_true',
                        default=False,
                        help="whether render position npy points")
    # Path adjustment
    parser.add_argument(
        "--offset", type=str, default="0,0,0", help="Center point to rotate around (only if not --traj)"
    )
    parser.add_argument("--radius", type=float, default=0.85, help="Radius of orbit (only if not --traj)")
    parser.add_argument(
        "--elevation",
        type=float,
        default=-5.0,
        help="Elevation of orbit in deg, negative is above",
    )
    parser.add_argument(
        "--elevation2",
        type=float,
        default=-10.0,
        help="Max elevation, only for spiral",
    )
    parser.add_argument(
        "--vec_up",
        type=str,
        default=None,
        help="up axis for camera views (only if not --traj);"
        "3 floats separated by ','; if not given automatically determined",
    )
    parser.add_argument(
        "--vert_shift",
        type=float,
        default=0.0,
        help="vertical shift by up axis"
    )
    parser.add_argument('--traj_type',
                    choices=['spiral', 'circle'],
                    default='spiral',
                    help="Render a spiral (doubles length, using 2 elevations), or just a cirle")
    parser.add_argument(
                    "--width", "-W", type=float, default=None, help="Rendering image width (only if not --traj)"
                            )
    parser.add_argument(
                        "--height", "-H", type=float, default=None, help="Rendering image height (only if not --traj)"
                                )
    parser.add_argument(
        "--num_views", "-N", type=int, default=600,
        help="Number of frames to render"
    )

    # Camera adjustment
    parser.add_argument('--crop',
                        type=float,
                        default=1.0,
                        help="Crop (0, 1], 1.0 = full image")

    # Foreground/background only
    parser.add_argument('--nofg',
                        action='store_true',
                        default=False,
                        help="Do not render foreground (if using BG model)")
    parser.add_argument('--nobg',
                        action='store_true',
                        default=False,
                        help="Do not render background (if using BG model)")

    # Random debugging features
    parser.add_argument('--blackbg',
                        action='store_true',
                        default=False,
                        help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
    parser.add_argument('--ray_len',
                        action='store_true',
                        default=False,
                        help="Render the ray lengths")

    args = parser.parse_args()
    return args

def produce_opt_args():
    parser = argparse.ArgumentParser()
    config_util.define_common_args(parser)


    #### ARF parameters
    parser.add_argument("--init_ckpt", type=str, default="", help="initial checkpoint to load")
    parser.add_argument("--style", type=str, help="path to style image")
    parser.add_argument("--tmpl", type=str, help="path to tmpl image")
    parser.add_argument("--match_folder", type=str, help="path to the source domain")
    parser.add_argument("--n_samples", type=int, default=50000, help="path to the source domain")

    # parser.add_argument("--pr_folder", type=str, help="dir to photo-realistic image")
    parser.add_argument("--content_weight", type=float, default=5e-3, help="content loss weight")
    parser.add_argument("--img_tv_weight", type=float, default=1, help="image tv loss weight")
    parser.add_argument(
        "--vgg_block",
        type=int,
        default=2,
        help="vgg block for extracting feature maps",
    )
    parser.add_argument(
        "--vgg_blocks",
        type=str,
        default=2,
        help="vgg block for extracting feature maps",
    )
    parser.add_argument(
        "--reset_basis_dim",
        type=int,
        default=1,
        help="whether to reset the number of spherical harmonics basis to this specified number",
    )
    parser.add_argument(
        "--mse_num_epoches",
        type=int,
        default=2,
        help="epoches for mse loss optimization",
    )
    parser.add_argument(
        "--nnfm_num_epoches",
        type=int,
        default=10,
        help="epoches for running style transfer",
    )
    parser.add_argument("--no_pre_ct", action="store_true", default=False)
    parser.add_argument("--no_post_ct", action="store_true", default=False)
    # ARF

    group = parser.add_argument_group("general")
    group.add_argument(
        "--train_dir",
        "-t",
        type=str,
        default="ckpt",
        help="checkpoint and logging directory",
    )

    group.add_argument(
        "--reso",
        type=str,
        default="[[256, 256, 256], [512, 512, 512]]",
        help="List of grid resolution (will be evaled as json);"
        "resamples to the next one every upsamp_every iters, then "
        + "stays at the last one; "
        + "should be a list where each item is a list of 3 ints or an int",
    )

    group.add_argument(
        "--upsamp_every",
        type=int,
        default=3 * 12800,
        help="upsample the grid every x iters",
    )
    group.add_argument("--init_iters", type=int, default=0, help="do not upsample for first x iters")
    group.add_argument(
        "--upsample_density_add",
        type=float,
        default=0.0,
        help="add the remaining density by this amount when upsampling",
    )

    group.add_argument(
        "--basis_type",
        choices=["sh", "3d_texture", "mlp"],
        default="sh",
        help="Basis function type",
    )

    group.add_argument(
        "--basis_reso",
        type=int,
        default=32,
        help="basis grid resolution (only for learned texture)",
    )
    group.add_argument("--sh_dim", type=int, default=9, help="SH/learned basis dimensions (at most 10)")

    group.add_argument(
        "--mlp_posenc_size",
        type=int,
        default=4,
        help="Positional encoding size if using MLP basis; 0 to disable",
    )
    group.add_argument("--mlp_width", type=int, default=32, help="MLP width if using MLP basis")

    group.add_argument(
        "--background_nlayers",
        type=int,
        default=0,  # 32,
        help="Number of background layers (0=disable BG model)",
    )
    group.add_argument("--background_reso", type=int, default=512, help="Background resolution")


    group = parser.add_argument_group("optimization")
    group.add_argument(
        "--n_iters",
        type=int,
        default=10 * 12800,
        help="total number of iters to optimize for",
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        # 100000,
        #      2000,
        help="batch size",
    )


    # TODO: make the lr higher near the end
    group.add_argument(
        "--sigma_optim",
        choices=["sgd", "rmsprop"],
        default="rmsprop",
        help="Density optimizer",
    )
    group.add_argument("--lr_sigma", type=float, default=3e1, help="SGD/rmsprop lr for sigma")
    group.add_argument("--lr_sigma_final", type=float, default=5e-2)
    group.add_argument("--lr_sigma_decay_steps", type=int, default=250000)
    group.add_argument(
        "--lr_sigma_delay_steps",
        type=int,
        default=15000,
        help="Reverse cosine steps (0 means disable)",
    )
    group.add_argument("--lr_sigma_delay_mult", type=float, default=1e-2)  # 1e-4)#1e-4)


    group.add_argument("--sh_optim", choices=["sgd", "rmsprop"], default="rmsprop", help="SH optimizer")
    group.add_argument("--lr_sh", type=float, default=1e-2, help="SGD/rmsprop lr for SH")
    group.add_argument("--lr_sh_final", type=float, default=5e-6)
    group.add_argument("--lr_sh_decay_steps", type=int, default=250000)
    group.add_argument(
        "--lr_sh_delay_steps",
        type=int,
        default=0,
        help="Reverse cosine steps (0 means disable)",
    )
    group.add_argument("--lr_sh_delay_mult", type=float, default=1e-2)

    group.add_argument(
        "--lr_fg_begin_step",
        type=int,
        default=0,
        help="Foreground begins training at given step number",
    )

    # BG LRs
    group.add_argument(
        "--bg_optim",
        choices=["sgd", "rmsprop"],
        default="rmsprop",
        help="Background optimizer",
    )
    group.add_argument("--lr_sigma_bg", type=float, default=3e0, help="SGD/rmsprop lr for background")
    group.add_argument(
        "--lr_sigma_bg_final",
        type=float,
        default=3e-3,
        help="SGD/rmsprop lr for background",
    )
    group.add_argument("--lr_sigma_bg_decay_steps", type=int, default=250000)
    group.add_argument(
        "--lr_sigma_bg_delay_steps",
        type=int,
        default=0,
        help="Reverse cosine steps (0 means disable)",
    )
    group.add_argument("--lr_sigma_bg_delay_mult", type=float, default=1e-2)

    group.add_argument("--lr_color_bg", type=float, default=1e-1, help="SGD/rmsprop lr for background")
    group.add_argument(
        "--lr_color_bg_final",
        type=float,
        default=5e-6,  # 1e-4,
        help="SGD/rmsprop lr for background",
    )
    group.add_argument("--lr_color_bg_decay_steps", type=int, default=250000)
    group.add_argument(
        "--lr_color_bg_delay_steps",
        type=int,
        default=0,
        help="Reverse cosine steps (0 means disable)",
    )
    group.add_argument("--lr_color_bg_delay_mult", type=float, default=1e-2)
    # END BG LRs

    group.add_argument(
        "--basis_optim",
        choices=["sgd", "rmsprop"],
        default="rmsprop",
        help="Learned basis optimizer",
    )
    group.add_argument("--lr_basis", type=float, default=1e-6, help="SGD/rmsprop lr for SH")  # 2e6,
    group.add_argument("--lr_basis_final", type=float, default=1e-6)
    group.add_argument("--lr_basis_decay_steps", type=int, default=250000)
    group.add_argument(
        "--lr_basis_delay_steps",
        type=int,
        default=0,  # 15000,
        help="Reverse cosine steps (0 means disable)",
    )
    group.add_argument("--lr_basis_begin_step", type=int, default=0)  # 4 * 12800)
    group.add_argument("--lr_basis_delay_mult", type=float, default=1e-2)

    group.add_argument("--rms_beta", type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument("--print_every", type=int, default=20, help="print every")
    group.add_argument("--save_every", type=int, default=1, help="save every x epochs")
    group.add_argument("--eval_every", type=int, default=1, help="evaluate every x epochs")

    group.add_argument("--init_sigma", type=float, default=0.1, help="initialization sigma")
    group.add_argument("--init_sigma_bg", type=float, default=0.1, help="initialization sigma (for BG)")

    # Extra logging
    group.add_argument("--log_mse_image", action="store_true", default=False)
    group.add_argument("--log_depth_map", action="store_true", default=False)
    group.add_argument(
        "--log_depth_map_use_thresh",
        type=float,
        default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term",
    )


    group = parser.add_argument_group("misc experiments")
    group.add_argument(
        "--thresh_type",
        choices=["weight", "sigma"],
        default="weight",
        help="Upsample threshold type",
    )
    group.add_argument(
        "--weight_thresh",
        type=float,
        default=0.0005 * 512,
        #  default=0.025 * 512,
        help="Upsample weight threshold; will be divided by resulting z-resolution",
    )
    group.add_argument("--density_thresh", type=float, default=5.0, help="Upsample sigma threshold")
    group.add_argument(
        "--background_density_thresh",
        type=float,
        default=1.0 + 1e-9,
        help="Background sigma threshold for sparsification",
    )
    group.add_argument(
        "--max_grid_elements",
        type=int,
        default=44_000_000,
        help="Max items to store after upsampling " "(the number here is given for 22GB memory)",
    )

    group.add_argument(
        "--tune_mode",
        action="store_true",
        default=False,
        help="hypertuning mode (do not save, for speed)",
    )
    group.add_argument(
        "--tune_nosave",
        action="store_true",
        default=False,
        help="do not save any checkpoint even at the end",
    )


    group = parser.add_argument_group("losses")
    # Foreground TV
    group.add_argument("--lambda_tv", type=float, default=1e-5)
    group.add_argument("--tv_sparsity", type=float, default=0.01)
    group.add_argument(
        "--tv_logalpha",
        action="store_true",
        default=False,
        help="Use log(1-exp(-delta * sigma)) as in neural volumes",
    )

    group.add_argument("--lambda_tv_sh", type=float, default=1e-3)
    group.add_argument("--tv_sh_sparsity", type=float, default=0.01)

    group.add_argument("--lambda_tv_lumisphere", type=float, default=0.0)  # 1e-2)#1e-3)
    group.add_argument("--tv_lumisphere_sparsity", type=float, default=0.01)
    group.add_argument("--tv_lumisphere_dir_factor", type=float, default=0.0)

    group.add_argument("--tv_decay", type=float, default=1.0)

    group.add_argument("--lambda_l2_sh", type=float, default=0.0)  # 1e-4)
    group.add_argument(
        "--tv_early_only",
        type=int,
        default=1,
        help="Turn off TV regularization after the first split/prune",
    )

    group.add_argument(
        "--tv_contiguous",
        type=int,
        default=1,
        help="Apply TV only on contiguous link chunks, which is faster",
    )
    # End Foreground TV

    group.add_argument(
        "--lambda_sparsity",
        type=float,
        default=0.0,
        help="Weight for sparsity loss as in SNeRG/PlenOctrees " + "(but applied on the ray)",
    )
    group.add_argument(
        "--lambda_beta",
        type=float,
        default=0.0,
        help="Weight for beta distribution sparsity loss as in neural volumes",
    )


    # Background TV
    group.add_argument("--lambda_tv_background_sigma", type=float, default=1e-2)
    group.add_argument("--lambda_tv_background_color", type=float, default=1e-2)

    group.add_argument("--tv_background_sparsity", type=float, default=0.01)
    # End Background TV

    # Basis TV
    group.add_argument(
        "--lambda_tv_basis",
        type=float,
        default=0.0,
        help="Learned basis total variation loss",
    )
    # End Basis TV

    group.add_argument("--weight_decay_sigma", type=float, default=1.0)
    group.add_argument("--weight_decay_sh", type=float, default=1.0)

    group.add_argument("--lr_decay", action="store_true", default=True)

    group.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of training images. Defaults to use all avaiable.",
    )

    group.add_argument(
        "--nosphereinit",
        action="store_true",
        default=False,
        help="do not start with sphere bounds (please do not use for 360)",
    )
    args = parser.parse_args()
    config_util.maybe_merge_config_file(args, allow_invalid=True)
    return args

def produce_args():
    parser = argparse.ArgumentParser()
    config_util.define_common_args(parser)
    parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
    parser.add_argument('--train', action='store_true', default=False, help='render train set')
    parser.add_argument('--render_path',
                        action='store_true',
                        default=False,
                        help="Render path instead of test images (no metrics will be given)")
    parser.add_argument('--out_dir',
                        default="./g1_generalized_pipeline/",
                        help="the default output dir")
    parser.add_argument('--timing',
                        action='store_true',
                        default=False,
                        help="Run only for timing (do not save images or use LPIPS/SSIM; "
                        "still computes PSNR to make sure images are being generated)")
    parser.add_argument('--no_lpips',
                        action='store_true',
                        default=False,
                        help="Disable LPIPS (faster load)")
    parser.add_argument('--no_vid',
                        action='store_true',
                        default=False,
                        help="Disable video generation")
    parser.add_argument('--no_imsave',
                        action='store_true',
                        default=False,
                        help="Disable image saving (can still save video; MUCH faster)")
    parser.add_argument('--fps',
                        type=int,
                        default=30,
                        help="FPS of video")
    parser.add_argument('--render_depth',
                        action='store_true',
                        default=False,
                        help="whether render depth image")
    parser.add_argument('--render_xyz',
                        action='store_true',
                        default=False,
                        help="whether render position npy points")

    # Camera adjustment
    parser.add_argument('--crop',
                        type=float,
                        default=1.0,
                        help="Crop (0, 1], 1.0 = full image")

    # Foreground/background only
    parser.add_argument('--nofg',
                        action='store_true',
                        default=False,
                        help="Do not render foreground (if using BG model)")
    parser.add_argument('--nobg',
                        action='store_true',
                        default=False,
                        help="Do not render background (if using BG model)")

    # Random debugging features
    parser.add_argument('--blackbg',
                        action='store_true',
                        default=False,
                        help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
    parser.add_argument('--ray_len',
                        action='store_true',
                        default=False,
                        help="Render the ray lengths")
    args = parser.parse_args()
    config_util.maybe_merge_config_file(args, allow_invalid=True)
    return args