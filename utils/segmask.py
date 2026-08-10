import os
import numpy as np
import pandas as pd

STRUCTS = ['Left Lung', 'Right Lung', 'Heart']

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'chexmask', 'ChestX-Ray8.csv')


def decode_rle(rle, out_size=256, source_size=1024):
    """Decode a CheXmask RLE string (start, count pairs, 1-indexed, row-major).

    :param rle: str. Space-separated start/count pairs at source_size x source_size.
    :param out_size: int. The mask is resized to out_size x out_size (nearest).
    :param source_size: int. Native mask resolution.
    :return: np.ndarray of shape (out_size, out_size), dtype uint8.
    """
    if isinstance(rle, str) and len(rle.strip()) == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    a = np.array(rle.split(), dtype=int)
    starts = a[0::2] - 1
    lengths = a[1::2]
    mask = np.zeros(source_size * source_size, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        mask[s:min(s + l, source_size * source_size)] = 1
    mask = mask.reshape(source_size, source_size)

    if out_size != source_size:
        from PIL import Image
        mask = (Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (out_size, out_size), Image.NEAREST))
        mask = np.asarray(mask) > 127
    return mask.astype(np.uint8)


def load_mask(row, out_size=256):
    """Compose the three structures of one CSV row into an (3, out_size, out_size) uint8 array."""
    return np.stack([decode_rle(row[s], out_size=out_size) for s in STRUCTS])


class CheXmaskFrame:
    """Lazily decodes and caches CheXmask ground-truth masks by Image Index."""

    def __init__(self, csv_path=DEFAULT_CSV):
        self.path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df = self.df.set_index('Image Index')
        self.cache = {}

    def get(self, image_index, out_size=256):
        if image_index not in self.df.index:
            return None
        if image_index not in self.cache:
            row = self.df.loc[image_index]
            self.cache[image_index] = load_mask(row, out_size=out_size)
        return self.cache[image_index]