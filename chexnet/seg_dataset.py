import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from utils.segmask import CheXmaskFrame


class SegDataset(Dataset):
    """Grayscale chest X-rays paired with CheXmask ground truth (Left/Right Lung, Heart, 256x256).

    :param fold: 'train', 'val' or 'test', matched to the NIH official split in chexnet/nih_labels.csv.
    :param image_path: Root folder holding the .png images.
    :param subsample: Optional cap on the number of loaded samples (for fast smoke runs).
    """

    def __init__(self, fold, image_path, subsample=0):
        self.fold = fold
        self.image_path = image_path
        self.frame = CheXmaskFrame()

        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      'chexnet', 'nih_labels.csv'))
        df = df[df['fold'] == fold]
        df = df.set_index('Image Index')
        # Restrict to rows that actually have masks (all NIH images should).
        df = df[df.index.isin(self.frame.df.index)]
        self.df = df
        if subsample:
            self.df = self.df.sample(n=subsample, random_state=42)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.index[idx]
        img = Image.open(os.path.join(self.image_path, name)).convert('L')
        img = img.resize((256, 256), Image.BILINEAR)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5

        mask = self.frame.get(name, out_size=256)
        mask = mask.astype(np.float32)

        img = np.expand_dims(img, 0)
        return img, mask, name