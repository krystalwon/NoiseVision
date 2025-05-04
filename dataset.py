import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision.tv_tensors import BoundingBoxes, Mask
from PIL import Image, ImageDraw

class SpectrogramDataset(Dataset):
    def __init__(self, keys, df, img_map, cls2idx, transforms=None):
        super().__init__()
        self.keys = keys
        self.df = df
        self.img_map = img_map
        self.cls2idx = cls2idx
        self.transforms = transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        shapes = self.df.loc[key]['shapes']
        img = Image.open(self.img_map[key]).convert('RGB')

        # labels
        labels = torch.tensor([self.cls2idx[s['label']] for s in shapes], dtype=torch.int64)
        # masks
        masks = []
        for s in shapes:
            mask = Image.new('L', img.size, 0)
            ImageDraw.Draw(mask).polygon([tuple(p) for p in s['points']], fill=1)
            masks.append(torch.from_numpy(np.array(mask)).bool())
        masks = Mask(torch.stack(masks))
        # boxes
        boxes = BoundingBoxes(
            data=torchvision.ops.masks_to_boxes(masks),
            format='xyxy', canvas_size=img.size[::-1]
        )

        target = {'masks': masks, 'boxes': boxes, 'labels': labels}
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
    
def move_data_to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [move_data_to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    elif hasattr(data, "to"):
        return data.to(device)
    else:
        return data

def collate_fn(batch):
    return tuple(zip(*batch))