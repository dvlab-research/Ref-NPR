import imageio
import torch
import torchvision
import torch.nn.functional as F
from icecream import ic

def ModifyMap(Style, Input, gmin, gmax):
    Gain = torch.div(Style, Input+1e-4)
    Gain = torch.clamp(Gain, min=gmin, max=gmax)
    return Input*Gain 

def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best


def nn_feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def nn_feat_replace_cond(tmpl, a, b):
    # feature matching with condition.
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()
    # n3, c, _, _ = tmpl.size()

    # assert (n == 1) and (n2 == 1) and (h == h2) and (w == w2) # JULIAN: actually no need.

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    tmpl_flat = tmpl.view(n, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    z_bests = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], tmpl_flat[i : i + 1])
        z_bests.append(z_best)
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new, torch.cat(z_bests, 0)

def get_nn_feat_relation(tmpl, a):
    # feature matching with condition.
    n, c, h, w = a.size()
    # n3, c, _, _ = tmpl.size()

    a_flat = a.view(n, c, -1)
    tmpl_flat = tmpl.view(n, c, -1)

    z_new = []
    z_bests = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], tmpl_flat[i : i + 1])
        z_bests.append(z_best)
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
    return torch.cat(z_bests, 0)

def get_patch_shuf_conv(patch_size=4):
    conv_layer = torch.nn.Conv2d(3, 3*patch_size*patch_size, 
                                kernel_size=patch_size, stride=patch_size, bias=False)
    for param in conv_layer.parameters():
        # print(param.size())
        
        out_size, in_size, k, _ = param.shape
        param = torch.zeros_like(param)
        for i in range(out_size):
            param[i][i%3][((i//3)%16)//4][(i//3)%4] = 1
        return param

import math
def nn_color_patch(top_color_centers, outputs, relation, vis=False):
    b,_,h,w = outputs.shape
    _, hw  = relation.shape
    patch_size = int(math.sqrt(h*w / hw))
    weight = get_patch_shuf_conv(patch_size).cuda()
    conv_res = torch.nn.functional.conv2d(outputs, weight, None, stride=patch_size).view(b, patch_size**2, 3, -1)

    # ic(conv_res, conv_res.shape)
    for _ in range(b):
        c = top_color_centers.shape[0]*top_color_centers.shape[1]
        # TODO: index relation here for multi-ref.
        relation_i = relation.unsqueeze(1).repeat(1, c, 1)
        # [SHAPE] top_color_centers: [5, 3, 40000] conv_res: [1, 16, 3, 40000]
        color_info = torch.gather(top_color_centers.view(1, c, -1), 2, relation_i)[0]
        color_info = color_info.view(c//3, 3, -1) # [SHAPE] 5, 3, 200*200

        temp_loss = torch.ones_like(torch.sum(conv_res[0].view(patch_size**2, 3, -1), dim=1))*3

        # JULIAN: calculate the nearest neighbor.
        for center in range(color_info.shape[0]):
            temp_centorid = color_info[center].unsqueeze(0).repeat(patch_size**2, 1, 1)
            temp_loss = torch.min(temp_loss, torch.sum(torch.nn.functional.mse_loss(temp_centorid, conv_res[0], reduce=False), dim=1))

    # ic(temp_loss.dtype)
    # ic(torch.mean(temp_loss))
    return 1*torch.mean(temp_loss)

def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

def cos_loss_match_norm(a, b):
    bs,c,h,w = a.shape
    a = a.view(bs, c, -1)
    b = b.view(bs, c, -1)
    # ic(torch.mean(b, dim=-1).shape)
    a = a - torch.mean(b, dim=-1).unsqueeze(2).repeat(1, 1, h*w)
    b = b - torch.mean(b, dim=-1).unsqueeze(2).repeat(1, 1, h*w)
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

def cos_loss_norm(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    # h, w = a_norm.shape[2:]
    # print(cos_d.mean(), F.mse_loss(a_norm, b_norm), a_norm.shape)
    return cos_d.mean() + F.mse_loss(a_norm, b_norm)*1e-4

def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G

def argmin_cos_distance_thre(a, b, thre=0.6, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)
        
        z_best_min, z_best_batch = torch.min(d_mat, 2)
        z_best_batch[z_best_min > thre] = -1
        # ic(d_mat.min())
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best

def get_nn_feat_relation_thre(tmpl, a):
    # feature matching with condition.
    n, c, h, w = a.size()

    a_flat = a.view(n, c, -1)
    tmpl_flat = tmpl.view(n, c, -1)

    z_new = []
    z_bests = []
    for i in range(n):
        z_best = argmin_cos_distance_thre(a_flat[i : i + 1], tmpl_flat[i : i + 1])
        z_bests.append(z_best)
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
    return torch.cat(z_bests, 0)

class NNFMLoss(torch.nn.Module):
    def __init__(self, device, gainmap=False, is_bn=False, layer=16):
        super().__init__()
        if layer==19:
            self.vgg = torchvision.models.vgg19(pretrained=True).eval().to(device)
            self.block_indexes = [[1, 3], [6, 8], [11, 13, 15, 17], [20, 22, 24, 26], [29, 31, 33, 35]]
        else:
            if not is_bn:
                self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
                self.block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
            else:
                self.vgg = torchvision.models.vgg16_bn(pretrained=True).eval().to(device)
                self.block_indexes = [[2, 5], [9, 12], [16, 19, 22], [26, 29, 32], [36, 39, 42]]

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.gainmap = gainmap
        self.gmin = 0.7
        self.gmax = 5.0


    def get_feats(self, x, layers=[]):
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
    
    def get_feats_by_blk(self, x, blocks=[]):
        all_layers = []
        for block in blocks:
            all_layers += self.block_indexes[block]
        x = self.normalize(x)
        final_ix = max(all_layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in all_layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def preload_golden_template(self, tmpl_imgs=None, tmpl_stys=None, blocks=[2,4]):
        # use this function to preload the nearest template content. 
        # (like nnfm, but matching on the same domain (content-content) domain)
        # pr_img: PhotoRealistic image.
        # JULIAN: 20221020 | a multi-reference version
        if not hasattr(self, "styl_feats_all"):
            blocks.sort()
            all_layers = []
            self.styl_feats_all = []
            self.tmpl_feats_all = []
            self.tmpl_stys = tmpl_stys
            self.tmpl_imgs = tmpl_imgs
            for block in blocks:
                all_layers += self.block_indexes[block]
            with torch.no_grad():
                for tmpl_sty, tmpl_img in zip(tmpl_stys, tmpl_imgs):
                    self.tmpl_feats_all += [self.get_feats(tmpl_img, all_layers)]
                    self.styl_feats_all += [self.get_feats(tmpl_sty.permute(2,0,1).unsqueeze(0), all_layers)]
                for layer in range(len(self.styl_feats_all[0])):
                    self.styl_feats_all[0][layer] = torch.concat([self.styl_feats_all[i][layer] for i in range(len(tmpl_stys))], dim=2)
                    self.tmpl_feats_all[0][layer] = torch.concat([self.tmpl_feats_all[i][layer] for i in range(len(tmpl_stys))], dim=2)

                self.styl_feats_all = self.styl_feats_all[0]
                self.tmpl_feats_all = self.tmpl_feats_all[0]
        else:
            blocks.sort()
            all_layers = []
            self.tmpl_feats_all = []
            self.tmpl_imgs = tmpl_imgs
            for block in blocks:
                all_layers += self.block_indexes[block]
            with torch.no_grad():
                for tmpl_img in tmpl_imgs:
                    self.tmpl_feats_all += [self.get_feats(tmpl_img, all_layers)]
                for layer in range(len(self.tmpl_feats_all[0])):
                    self.tmpl_feats_all[0][layer] = torch.concat([self.tmpl_feats_all[i][layer] for i in range(len(tmpl_stys))], dim=2)
                self.tmpl_feats_all = self.tmpl_feats_all[0]
        
    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
    ):
        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss', 'tcm_loss', 'color_patch', 'online_tmp_loss']
            # multi-view correspondence loss.
        # block_indexes = self.block_indexes

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += self.block_indexes[block]

        x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            if hasattr(self, 'styl_feats_all'):
                s_feats_all = self.styl_feats_all
            else:
                s_feats_all = self.get_feats(styles, all_layers)
            if "content_loss" in loss_names or "tcm_loss" in loss_names:
                content_feats_all = self.get_feats(contents, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        # JULIAN: the pre-processing part for coarse color transformation.
        if 'tcm_loss' in loss_names:
            coarse_style_flat = []
            down_fact = 16
            for tmpl_sty in self.tmpl_stys:
                h_sty, w_sty  = tmpl_sty.shape[:2]
                coarse_style_flat.append(F.interpolate(tmpl_sty.unsqueeze(0).permute(0, 3, 1, 2),
                                                    (h_sty//down_fact, w_sty//down_fact), mode='bilinear', antialias=True, align_corners=True))
            coarse_style_flat =  torch.cat(coarse_style_flat, dim=-2)
            coarse_style_flat = coarse_style_flat.view(1, 3, -1)
            coarse_style_flat = torch.cat((coarse_style_flat, torch.FloatTensor([[[0], [0], [0]]]).cuda()), dim=-1)
            # outputs_flat = outputs
            coarse_out_flat = F.interpolate(outputs,
                                (h_sty//down_fact, w_sty//down_fact), mode='bilinear', antialias=True, align_corners=True).view(1, 3, -1)

        loss_dict = dict([(x, 0.) for x in loss_names])
        if len(blocks)==1:
            blocks += [-1]
            
        for block in blocks[:-1]:
            layers = self.block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if "nnfm_loss" in loss_names:
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                loss_dict["gram_loss"] += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)

            if "content_loss" in loss_names or "tcm_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                
            if "content_loss" in loss_names:
                if self.gainmap:
                    # print(content_feats)
                    content_feats = ModifyMap(s_feats, content_feats, self.gmin, self.gmax)
                    # print("modify gainmap", content_feats)
                loss_dict["content_loss"] += torch.mean((content_feats - x_feats) ** 2)

            if "tcm_loss" in loss_names:
                tmpl_feats = torch.cat([self.tmpl_feats_all[ix_map[ix]] for ix in layers], 1)
                target_feats, relation = nn_feat_replace_cond(tmpl_feats, content_feats, s_feats)
                loss_dict["tcm_loss"] += cos_loss(x_feats, target_feats)

            # JULIAN: keep updateing the tmpl image here.
            if "online_tmp_loss" in loss_names:
                tmpl_feats = torch.cat([self.tmpl_feats_all[ix_map[ix]] for ix in layers], 1)
                target_feats, relation = nn_feat_replace_cond(tmpl_feats, x_feats, s_feats)
                loss_dict["online_tmp_loss"] += cos_loss(x_feats, target_feats)*0.2


            if "color_patch" in loss_names:
                layers_last = self.block_indexes[blocks[-1]]
                
                # s_feats_last = torch.cat([s_feats_all[ix_map[ix]] for ix in layers_last], 1)
                tmpl_feats_last = torch.cat([self.tmpl_feats_all[ix_map[ix]] for ix in layers_last], 1)
                if "online_tmp_loss" not in loss_names:
                    content_feats_last = torch.cat([content_feats_all[ix_map[ix]] for ix in layers_last], 1)
                    relation_last = get_nn_feat_relation_thre(tmpl_feats_last, content_feats_last).repeat(1, 3, 1)
                else:
                    content_feats_last = torch.cat([x_feats_all[ix_map[ix]] for ix in layers_last], 1)
                    relation_last = get_nn_feat_relation_thre(tmpl_feats_last, content_feats_last).repeat(1, 3, 1)

                relation_last[relation_last < 0] = coarse_style_flat.shape[-1] - 1
                related_img = torch.gather(coarse_style_flat, 2, relation_last)
                loss_patch = (related_img - coarse_out_flat)**2
                loss_patch[relation_last == coarse_style_flat.shape[-1] - 1] = 0
                loss_patch = loss_patch.mean(dim=1)
                coarse_out_flat_mask = coarse_out_flat.mean(dim=1)
                # add a mask for the nerf dataset.
                loss_patch[coarse_out_flat_mask > 0.99] = 0
                loss_dict["color_patch"] = torch.mean(loss_patch)*5

        return loss_dict


""" VGG-16 Structure
Input image is [-1, 3, 224, 224]
-------------------------------------------------------------------------------
        Layer (type)               Output Shape         Param #     Layer index
===============================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792     
              ReLU-2         [-1, 64, 224, 224]               0               1
            Conv2d-3         [-1, 64, 224, 224]          36,928     
              ReLU-4         [-1, 64, 224, 224]               0               3
         MaxPool2d-5         [-1, 64, 112, 112]               0     
            Conv2d-6        [-1, 128, 112, 112]          73,856     
              ReLU-7        [-1, 128, 112, 112]               0               6
            Conv2d-8        [-1, 128, 112, 112]         147,584     
              ReLU-9        [-1, 128, 112, 112]               0               8
        MaxPool2d-10          [-1, 128, 56, 56]               0     
           Conv2d-11          [-1, 256, 56, 56]         295,168     
             ReLU-12          [-1, 256, 56, 56]               0              11
           Conv2d-13          [-1, 256, 56, 56]         590,080     
             ReLU-14          [-1, 256, 56, 56]               0              13
           Conv2d-15          [-1, 256, 56, 56]         590,080     
             ReLU-16          [-1, 256, 56, 56]               0              15
        MaxPool2d-17          [-1, 256, 28, 28]               0     
           Conv2d-18          [-1, 512, 28, 28]       1,180,160     
             ReLU-19          [-1, 512, 28, 28]               0              18
           Conv2d-20          [-1, 512, 28, 28]       2,359,808     
             ReLU-21          [-1, 512, 28, 28]               0              20
           Conv2d-22          [-1, 512, 28, 28]       2,359,808     
             ReLU-23          [-1, 512, 28, 28]               0              22
        MaxPool2d-24          [-1, 512, 14, 14]               0     
           Conv2d-25          [-1, 512, 14, 14]       2,359,808     
             ReLU-26          [-1, 512, 14, 14]               0              25
           Conv2d-27          [-1, 512, 14, 14]       2,359,808     
             ReLU-28          [-1, 512, 14, 14]               0              27
           Conv2d-29          [-1, 512, 14, 14]       2,359,808    
             ReLU-30          [-1, 512, 14, 14]               0              29
        MaxPool2d-31            [-1, 512, 7, 7]               0    
===============================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.39  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(2): ReLU(inplace=True)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(5): ReLU(inplace=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(9): ReLU(inplace=True)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(12): ReLU(inplace=True)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(16): ReLU(inplace=True)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(19): ReLU(inplace=True)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(26): ReLU(inplace=True)
    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(32): ReLU(inplace=True)
    (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(36): ReLU(inplace=True)
    (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(39): ReLU(inplace=True)
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    *(42): ReLU(inplace=True)
    (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
Params size (MB): 56.13
Estimated Total Size (MB): 275.10
----------------------------------------------------------------
"""

"""
VGG-16-bn Structure

"""


if __name__ == '__main__':
    device = torch.device('cuda:0')
    nnfm_loss_fn = NNFMLoss(device)
    fake_output = torch.rand(1, 3, 256, 256).to(device)
    fake_style = torch.rand(1, 3, 256, 256).to(device)
    fake_content = torch.rand(1, 3, 256, 256).to(device)

    loss = nnfm_loss_fn(outputs=fake_output, styles=fake_style, contents=fake_content, loss_names=["nnfm_loss", "content_loss", "gram_loss"])
    ic(loss)

    fake_image_set = torch.rand(10, 256, 256, 3).to(device)
    fake_style = torch.rand(256, 256, 3).to(device)
    fake_image_set_new, color_tf = match_colors_for_image_set(fake_image_set, fake_style)
    ic(fake_image_set_new.shape, color_tf.shape)
