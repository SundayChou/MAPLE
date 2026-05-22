import torch

from PIL import Image
from einops import rearrange
from torchvision import transforms

from .hipt_features import *
from .hipt_utils import get_vit256, get_vit4k


Image.MAX_IMAGE_PIXELS = None
torch.multiprocessing.set_sharing_strategy('file_system')


class HIPT_4K(torch.nn.Module):
    def __init__(self, 
        model256_path=None,
        model4k_path=None, 
        device256=torch.device('cuda:0'), 
                device4k=torch.device('cuda:0')):
        super().__init__()
        self.model256 = get_vit256(pretrained_weights=model256_path).to(device256)
        self.model4k = get_vit4k(pretrained_weights=model4k_path).to(device4k)
        self.device256 = device256
        self.device4k = device4k

    def forward(self, x):
        return self.forward_all(x)[0]

    def forward_all(self, x):
        features_cls256, features_sub256 = self.forward_all256(x)
        features_cls4k, features_sub4k = self.forward_all4k(features_cls256)

        return features_cls4k, features_sub4k, features_sub256

    def forward_all256(self, x):
        batch_256, w_256, h_256 = self.prepare_img_tensor(x)
        batch_256 = batch_256.unfold(2, 256, 256).unfold(3, 256, 256)
        batch_256 = rearrange(batch_256, 'b c p1 p2 w h -> (b p1 p2) c w h')

        features_cls256 = []
        features_sub256 = []
        for mini_bs in range(0, batch_256.shape[0], 256):
            minibatch_256 = batch_256[mini_bs:mini_bs+256].to(self.device256, non_blocking=True)
            fea_all256 = self.model256.forward_all(minibatch_256).cpu()
            fea_cls256 = fea_all256[:, 0]
            fea_sub256 = fea_all256[:, 1:]
            features_cls256.append(fea_cls256)
            features_sub256.append(fea_sub256)

        features_cls256 = torch.vstack(features_cls256)
        features_sub256 = torch.vstack(features_sub256)
        features_cls256 = features_cls256.reshape(w_256, h_256, 384).transpose(0,1).transpose(0,2).unsqueeze(dim=0)
        features_sub256 = features_sub256.reshape(w_256, h_256, 16, 16, 384).permute(4, 0, 1, 2, 3).unsqueeze(dim=0)

        return features_cls256, features_sub256

    def forward_all4k(self, features_cls256):
        __, __, w_256, h_256 = features_cls256.shape
        features_cls256 = features_cls256.to(self.device4k, non_blocking=True)
        features_all4k = self.model4k.forward_all(features_cls256)
        features_cls4k = features_all4k[:, 0]
        features_sub4k = features_all4k[:, 1:]
        features_sub4k = features_sub4k.reshape(1, w_256, h_256, 192).permute(0, 3, 1, 2)

        return features_cls4k, features_sub4k

    def prepare_img_tensor(self, img: torch.Tensor, patch_size=256):
        make_divisble = lambda l, patch_size: (l - (l % patch_size))
        __, __, w, h = img.shape
        load_size = make_divisble(w, patch_size), make_divisble(h, patch_size)
        w_256, h_256 = w // patch_size, h // patch_size
        img_new = transforms.CenterCrop(load_size)(img)

        return img_new, w_256, h_256