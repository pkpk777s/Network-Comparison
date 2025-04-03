from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms.functional as tv
import kornia
import kornia.feature as KF

class Preprocessor:
    def __init__(self, scales=[0.5, 1.0, 1.5], blur_radius=2):
        self.scales = scales
        self.blur_radius = blur_radius

    def multi_scale(self, image):
        multi = {}
        features = {}
        w, h = image.size
        harris_detector = KF.HarrisDetector(k=0.04, window_size=3, threshold=1e-6)
        for scale in self.scales:
            new_size = (int(h * scale), int(w * scale))
            resized_img = tv.resize(image, new_size)
            multi[f'scale_{scale}'] = resized_img

            img_tensor = tv.to_tensor(resized_img).unsqueeze(0)
            gray = kornia.color.rgb_to_grayscale(img_tensor)  
            response = harris_detector(gray)
            keypoints = KF.get_keypoints_from_response(response, num_points=200)
            patch_size = 32
            patches = KF.extract_patches_from_tensor(gray, keypoints, patch_size)
            descriptors = patches.view(patches.shape[0], patches.shape[1], -1)
            
            features[f'scale_{scale}'] = {
                "keypoints": keypoints,
                "descriptors": descriptors
            }
            
        return multi, features

    def high_low_frequency(self, image):
        low_freq = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        
        image_np = np.array(image, dtype=np.float32)
        low_np = np.array(low_freq, dtype=np.float32)
        
        high_np = image_np - low_np + 128
        high_np = np.clip(high_np, 0, 255).astype(np.uint8)
        high_freq = Image.fromarray(high_np)
        
        return high_freq, low_freq

    def __call__(self, image):
        output = {}
        multi_scale_images, multi_scale_features = self.multi_scale(image)
        output['multi_scale_images'] = multi_scale_images
        output['multi_scale_features'] = multi_scale_features
        output['high_frequency'], output['low_frequency'] = self.high_low_frequency(image)
        return output

