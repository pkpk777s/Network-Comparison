import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class FaceDataset(Dataset):
    def __init__(self, base_dir, preload=False):

        self.base_dir = base_dir
        self.preload = preload 
        # preload: if True, load all images into memory, dont use it if less memory
        # could think about putting them in cpu later? 
        self.all_data = []
        
        for md in ['train', 'test', 'validation']:
            img_dir = os.path.join(self.base_dir, md)
            if not os.path.isdir(img_dir):
                continue
            image_files = [f for f in os.listdir(img_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for filename in image_files:
                full_path = os.path.join(img_dir, filename)
                # 'F_' images are Fake (False), 'R_' images are Real (True)
                label = False if filename.startswith('F_') else True
                if preload:
                    image = read_image(full_path)
                    self.all_data.append({'image': image, 'label': label, 'path': full_path, 'mode': md})
                else:
                    self.all_data.append({'path': full_path, 'label': label, 'mode': md})
        self.set_mode('train')
    
    def set_mode(self, mode):
        self.data = [entry for entry in self.all_data if entry['mode'] == mode]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        if 'image' in entry:
            return entry
        image = read_image(entry['path'])
        entry['image'] = image
        return entry

class FaceDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        self.dataset = dataset 
    
    def set(self, mode):
        "switch between train, test, and validation modes"
        self.dataset.set_mode(mode)

if __name__ == "__main__":
    "Example usage of the FaceDataset and FaceDataLoader"
    base_dir = "./data" # Change to relative path later
    dataset = FaceDataset(base_dir, preload=False)
    dataLoader = FaceDataLoader(dataset, batch_size=4, shuffle=True)
    dataLoader.set("train")
    print("Training Data:")
    for batch in dataLoader:
        print(batch)
        break

    dataLoader.set("test")
    print("Testing Data:")
    for batch in dataLoader:
        print(batch)
        break

    dataLoader.set("validation")
    print("Validation Data:")
    for batch in dataLoader:
        print(batch)
        break
