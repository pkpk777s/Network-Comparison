# Preprocessor.py
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms.functional as tv
import torch


class Preprocessor:
    def __init__(self, scales=[0.5, 1.0, 1.5], blur_radius=2):
        self.scales = scales
        self.blur_radius = blur_radius

    def multi_scale(self, image: Image.Image):
        """
        Generate resized images at each scale.
        Returns a dict mapping scale names to PIL.Image objects.
        """
        multi = {}
        w, h = image.size

        for scale in self.scales:
            new_size = (int(h * scale), int(w * scale))
            resized_img = tv.resize(image, new_size)
            multi[f'scale_{scale}'] = resized_img

        return multi

    def high_low_frequency(self, image: Image.Image):
        low_freq = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        image_np = np.array(image, dtype=np.float32)
        low_np = np.array(low_freq, dtype=np.float32)
        high_np = image_np - low_np + 128
        high_np = np.clip(high_np, 0, 255).astype(np.uint8)
        high_freq = Image.fromarray(high_np)

        return high_freq, low_freq

    def __call__(self, image: Image.Image):
        multi_imgs = self.multi_scale(image)
        high_freq, low_freq = self.high_low_frequency(image)
        return {
            'multi_scale_images': multi_imgs,
            'high_frequency': high_freq,
            'low_frequency': low_freq
        }

