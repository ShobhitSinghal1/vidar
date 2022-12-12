from vidar.utils.setup import setup_arch
from vidar.utils.config import cfg_has, read_config
from PIL import Image
import torchvision
import torch
import matplotlib.pyplot as plt

img_path = "data/Generic_kitti/2011_09_26/0000000000.png"  # "data/TritonEF/TritonEF/EF_sa-camera-1_1624708799868215.jpg"
cfg_path = "configs/overfit/generic_kitti.yaml"  # 'configs/papers/selfcalib/ucm_TritonEF.yaml'  #
checkpoint_path = '/data/vidar/checkpoints/2022-12-09_16h33m24s/models/046.ckpt'
cfg = read_config(cfg_path)
model = setup_arch(cfg.arch, checkpoint=checkpoint_path)

rgb = Image.open(img_path)
rgb = rgb.resize((640, 192))  #
rgb = torchvision.transforms.functional.pil_to_tensor(rgb)
rgb = torch.unsqueeze(rgb, 0)

depth = model.networks['depth'](rgb)

for i in range(4):
    di = depth['depths'][i].detach().numpy()
    plt.imshow(di[0,0])
    plt.savefig(f'{str(i)}.jpg')

print(model.networks['intrinsics'](rgb))