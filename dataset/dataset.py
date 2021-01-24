import os
import cv2
import albumentations as A
import albumentations.pytorch as Atorch

import torch
from torch.utils.data import Dataset


def load_dataset(dataset_path):
    imgs = []
    labels = []
    
    for label_idx, label_name in enumerate(os.listdir(dataset_path)):
        img_folder_path = os.path.join(dataset_path, label_name)             
        imgs_per_label = [os.path.join(img_folder_path, img).replace(os.sep, "/") for img in os.listdir(img_folder_path)]

        imgs.extend(imgs_per_label)
        labels.extend([label_idx]*len(imgs_per_label))

    assert len(imgs) == len(labels)

    labels = torch.LongTensor(labels)
    return imgs, labels

class ImageClassificationDataset(Dataset):
    def __init__(self, 
                 dataset_path,
                 phase
                 ):
        
        self.imgs, self.labels = load_dataset(dataset_path)

        if phase == "train":
            self.transform = A.Compose([A.RandomResizedCrop(224, 224, always_apply=True, p=1),
                                        A.HorizontalFlip(p=0.5),
                                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        Atorch.ToTensor()])
        elif phase == "validation":
            self.transform = A.Compose([A.Resize(256, 256), 
                            A.CenterCrop(224, 224),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            Atorch.ToTensor()])
    
    def __getitem__(self, index):

        img = cv2.imread(self.imgs[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        label = self.labels[index]

        return img, label

    def __len__(self):
        return len(self.imgs)