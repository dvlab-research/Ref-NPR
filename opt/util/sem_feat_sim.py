# use this file to calculate patchwise VGG similarity map.

from math import floor
import cv2
import torch
import torchvision
import imageio
import argparse
import numpy as np
import torch.nn.functional as F
from icecream import ic
from util import viridis_no_norm_cmap

parser = argparse.ArgumentParser()
parser.add_argument("--downrate", type=int, default=16, help="downsample_rate")
args = parser.parse_args()

class VGGPatchSim(torch.nn.Module):
    def __init__(self, device, args=None):
        super().__init__()

        self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.downrate = args.downrate

    def get_feats(self, x, layers=None):
        if layers is None:
            return []
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break
        return outputs

    def forward(self, img, style, layers=None):
        if layers is None:
            layers = [6, 11, 18, 25]
        h, w = img.shape[-2:]
        img_feats = self.get_feats(img, layers)
        style_feats = self.get_feats(style, layers)
        img_sim_mats = []
        style_sim_mats = []
        for i, (img_feat, style_feat) in enumerate(zip(img_feats, style_feats)):
            img_feat = F.interpolate(img_feat, (h // self.downrate, w // self.downrate), mode='bilinear')
            style_feat = F.interpolate(style_feat, (h // self.downrate, w // self.downrate), mode='bilinear')
            c = img_feat.shape[1]
            img_feat = F.normalize(img_feat[0].reshape(c, -1), dim=0)
            style_feat = F.normalize(style_feat[0].reshape(c, -1), dim=0)
            ic(f"level {i}", img_feat.shape, style_feat.shape)
            img_sim_mat = torch.mm(img_feat.T, img_feat)
            img_sim_mats += [img_sim_mat]
            style_sim_mat = torch.mm(style_feat.T, style_feat)
            style_sim_mats += [style_sim_mat]

        return img_sim_mats, style_sim_mats

    def visualize_patch(self, img, sim_mat, h_in, w_in, path="sim_vis.png"):
        h, w = img.shape[:2]
        feat_h, feat_w = h//self.downrate, w//self.downrate
        idx_h = floor(h_in*h/self.downrate)
        idx_w = floor(w_in*w/self.downrate)
        ic(idx_h, idx_w, sim_mat.shape)

        sim_mat = sim_mat[idx_h*feat_w + idx_w].view(feat_h, feat_w).unsqueeze(0).unsqueeze(0)
        sim_mat = F.interpolate(sim_mat, (h, w), mode='bilinear')[0][0].cpu().detach().numpy()
        sim_mat = viridis_no_norm_cmap(sim_mat)
        imageio.imwrite(path, (sim_mat*128 + img*128).astype(np.uint8))
        ic(sim_mat.shape)



if __name__ == '__main__':
    device = torch.device('cuda:0')
    patch_sim_fn = VGGPatchSim(device, args)
    img_path = "../../data/aligned_styles/imgs/0112.png"
    style_path = "../../data/styles/141.jpg"
    img_np = imageio.imread(img_path).astype(np.float32) / 255.0
    style_np = imageio.imread(style_path).astype(np.float32) / 255.0
    img = torch.from_numpy(img_np).to(device=device).unsqueeze(0).permute(0,3,1,2)
    style = torch.from_numpy(style_np).to(device=device).unsqueeze(0).permute(0,3,1,2)
    ic(img.shape, style.shape)
    img_sim_mats, style_sim_mats = patch_sim_fn(img, style)
    
    for i in range(len(img_sim_mats)):
        patch_sim_fn.visualize_patch(img_np, img_sim_mats[i], 0.1, 0.1, path=f"sim_vis_{i}.png")
        patch_sim_fn.visualize_patch(style_np, style_sim_mats[i], 0.1, 0.1, path=f"sty_vis_{i}.png")
    