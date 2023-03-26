# gainmap stylization module.
from operator import iconcat
import numpy as np
import torch
import torch.nn.functional as F
import os
import imageio
from nnfm_loss import NNFMLoss

# common parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
nnfm_loss_fn = NNFMLoss(device=device, gainmap=True, is_bn=False, layer=19)
imgs = []
style_imgs = []
transfer_iteration = 20
vgg_block = [1,2,3,4]
img_paths = sorted(os.listdir("../data/aligned_styles/imgs"))
style_paths = sorted(os.listdir("../data/aligned_styles/blue_flower_patch_res_wo_depth"))
out_dir = "../data/aligned_styles/blue_flower_patch_res_wo_depth_gainmap"
for img_path, style_path in zip(img_paths, style_paths):
    print(os.path.join("../data/aligned_styles/imgs", img_path), style_path)
    imgs.append(torch.from_numpy(
                imageio.imread(os.path.join("../data/aligned_styles/imgs", img_path))
                ).float().to(device=device)/255)
    style_imgs.append(torch.from_numpy(
                imageio.imread(os.path.join("../data/aligned_styles/blue_flower_patch_res_wo_depth", style_path), as_gray=False, pilmode="RGB")
                ).float().to(device=device)/255)

os.makedirs(out_dir, exist_ok = True)
for i, (rgb_gt, style_gt) in enumerate(zip(imgs, style_imgs)):
    print(i)
    stylized_gt = style_gt.clone().unsqueeze(0).permute(0,3,1,2).contiguous().to(device=device)
    #torch.rand(rgb_gt.shape).unsqueeze(0).permute(0,3,1,2).contiguous().to(device=device)
    #rgb_gt.clone().unsqueeze(0).permute(0,3,1,2).contiguous().to(device=device)
    stylized_gt = torch.nn.Parameter(stylized_gt, requires_grad=True)
    # ic("stylized_gt shape: ", stylized_gt.shape)
    style_img_optimizer = torch.optim.LBFGS([stylized_gt], lr=1.0)
    
    n_iter = 0
    while n_iter < transfer_iteration:
        def closure():
            style_img_optimizer.zero_grad()
            loss_dict = nnfm_loss_fn(
                stylized_gt,
                style_gt.permute(2, 0, 1).unsqueeze(0),
                blocks=vgg_block,
                loss_names=["gram_loss", "content_loss"],
                contents=rgb_gt.permute(2, 0, 1).unsqueeze(0)
            )
            # n_iter[0] += 1
            # normalize gram matrix loss.
            loss = loss_dict["gram_loss"] + 6e6*loss_dict["content_loss"]
            loss.backward(retain_graph=True)
            # print(loss_dict)
            # if n_iter[0] % 100 == 0:
            #     print(n_iter[0], loss_dict)
            return loss
        style_img_optimizer.step(closure)
        n_iter += 1

    imageio.imwrite(
        os.path.join(out_dir, f"data_{i}.png"),
        np.clip(stylized_gt[0].permute(1,2,0).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8),
    )
