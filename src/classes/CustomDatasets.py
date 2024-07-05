import os
import glob

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image
import xmltodict


class CustomVOCDataset(Dataset):
    def __init__(self, root, image_set, voc_classes, transforms=None):
        self.root = root
        self.voc_classes = voc_classes

        if image_set == "train":
            self.root = os.path.join(root, "train")
        elif image_set == "valid":
            self.root = os.path.join(root, "valid")
        else:
            raise NameError(f"в датасете нет папки {image_set}")
        self.transforms = transforms

        self.imgs = glob.glob(os.path.join(self.root, "*.jpg"))
        annotations_files = glob.glob((os.path.join(self.root, "*.xml")))

        # Инициализация аннотаций
        self.annotations = []
        for annotation in annotations_files:
            with open(annotation, 'r') as f:
                xml_data = f.read()
                xml_data = xmltodict.parse(xml_data)

                self.annotations.append(xml_data)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = read_image(img_path)
        img = tv_tensors.Image(img, dtype=torch.float32)

        target = self.annotations[idx]["annotation"]
        boxes = []
        labels = []
        if isinstance(target['object'], dict):
            obj = target['object']
            bbox = obj['bndbox']
            boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
            labels.append(self.voc_classes[obj['name']])
        else:
            for obj in target['object']:
                bbox = obj['bndbox']
                boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
                labels.append(self.voc_classes[obj['name']])

        # Перевод данных в Тензоры
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Формирование target
        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            self.transforms(img)
            self.transforms(target)

        return img, target
