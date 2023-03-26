import json
import os
from arguments import produce_opt_args
args = produce_opt_args()
data_dir = args.data_dir
with open(os.path.join(data_dir, "data_config.json")) as fp:
    style_dict = json.load(fp)
    
args.dataset_name   = style_dict["dataset_type"]
args.scene_name     = style_dict["scene_name"]

from icecream import ic
ic(args.dataset_name, args.scene_name)

if not os.path.exists(f"./exps/base_pr/ckpt_svox2/{args.dataset_name}/{args.scene_name}/ckpt.npz"):
    print(f"Did not find a PR checkpoint at ./exps/base_pr/ckpt_svox2/{args.dataset_name}/{args.scene_name}")
    import subprocess
    bashCommand = f"bash ./exps/base_pr/run_single.sh {args.dataset_name} {args.scene_name}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
else:
    print(f"Found the PR checkpoint at ./exps/base_pr/ckpt_svox2/{args.dataset_name}/{args.scene_name}")