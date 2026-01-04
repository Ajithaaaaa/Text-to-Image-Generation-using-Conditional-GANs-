# dataset.py
import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageCaptionDataset(Dataset):
    def __init__(self, images_dir, captions_file, img_size=64):
        self.images_dir = images_dir
        self.records = []
        # read csv
        with open(captions_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # try detect header
            first = next(reader)
            if len(first) >= 2 and first[0].lower().strip() == 'filename':
                # header exists, continue
                for row in reader:
                    if len(row) >= 2:
                        self.records.append((row[0].strip(), row[1].strip()))
            else:
                # first row was data
                if len(first) >= 2:
                    self.records.append((first[0].strip(), first[1].strip()))
                for row in reader:
                    if len(row) >= 2:
                        self.records.append((row[0].strip(), row[1].strip()))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # map to [-1,1]
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        filename, caption = self.records[idx]
        path = os.path.join(self.images_dir, filename)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, caption

def get_dataloader(images_dir, captions_file, img_size=64, batch_size=32, shuffle=True, num_workers=4):
    ds = ImageCaptionDataset(images_dir, captions_file, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
