import torch
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Precision, Recall, F1Score
from typing import Any, List, Dict, Tuple
from clearml import Task
from clearml import Logger as ClearMLLogger
import os


# Параметры логирования проекта
PROJECT_NAME = 'YourProjectName'
TASK_NAME = 'DoobpucheniyaModeliSSD'

# Импортируем ClearML Task
task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)

# Определение классов
VOC_CLASSES = {
    "backpack": 1,
    "barrel": 2,
    "bicycle": 3,
    "box": 4,
    "bunch-of-rocks": 5,
    "cardboard-box": 6,
    "dumpster": 7,
    "fallen-bicycle": 8,
    "fallen-tree": 9,
    "garbage-bag": 10,
    "handbag": 11,
    "metal-cart": 12,
    "metal-container": 13,
    "road-block": 14,
    "rock": 15,
    "sack": 16,
    "scooter": 17,
    "shovel": 18,
    "sport-bag": 19,
    "wooden-log": 20,
}


class VOCDataset(VOCDetection):
    def __init__(self, root, year="2012", image_set="train", transforms=None):
        super().__init__(root, year=year, image_set=image_set, transforms=transforms)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(idx)
        target = target['annotation']
        boxes = []
        labels = []
        for obj in target['object']:
            bbox = obj['bndbox']
            boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
            labels.append(VOC_CLASSES[obj['name']])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def get_transform(train):
    transforms = [F.to_tensor]
    return F.Compose(transforms)


class SSDLightning(pl.LightningModule):
    def __init__(self, num_classes):
        super(SSDLightning, self).__init__()
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)

        # Если не учитываем изображения с классом = background
        in_features = self.model.head.classification_head.conv[0].conv.in_channels
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.conv = torch.nn.Conv2d(
            in_features, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Если учитываем изображения с классом = background
        # self.model.head.classification_head.num_classes = num_classes

        # Метрики
        self.map_metric = MeanAveragePrecision()
        self.precision_metric = Precision()
        self.recall_metric = Recall()
        self.f1_metric = F1Score()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)

        # Обновление метрик
        preds = self.model(images)
        for pred, target in zip(preds, targets):
            self.map_metric.update(pred, target)
            self.precision_metric.update(pred['labels'], target['labels'])
            self.recall_metric.update(pred['labels'], target['labels'])
            self.f1_metric.update(pred['labels'], target['labels'])

        return losses

    def training_epoch_end(self, outputs):
        # Логирование метрик
        map50 = self.map_metric.compute()
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        f1 = self.f1_metric.compute()

        self.log("mAP50", map50)
        self.log("precision", precision)
        self.log("recall", recall)
        self.log("f1_score", f1)

        # Сброс метрик
        self.map_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]


def main():
    # Путь к данным
    data_dir = '/path/to/VOCdataset'

    # Определение датасетов и DataLoader
    train_dataset = VOCDataset(data_dir, year='2012', image_set='trainval', transforms=get_transform(train=True))
    val_dataset = VOCDataset(data_dir, year='2012', image_set='val', transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2,
                            collate_fn=lambda x: tuple(zip(*x)))

    # Инициализация модели
    model = SSDLightning(num_classes=20)  # 20 классов без учета фона

    # Определение логгера и коллбэков
    logger = TensorBoardLogger("tb_logs", name="ssdlite320")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=1, mode='min', verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping_callback = EarlyStopping(monitor="train_loss", patience=20)

    # Инициализация Trainer и запуск обучения
    trainer = pl.Trainer(
        max_epochs=10,
        logger=[logger, ClearMLLogger()],
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
        gpus=1 if torch.cuda.is_available() else 0,
    )
    trainer.fit(model, train_loader, val_loader)

    # Сохранение наилучших весов модели
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        task.upload_artifact('best_model', best_model_path)
        print(f'Лучшие веса модели сохранены в: {best_model_path}')

        # Сохранение метрик в ClearML
        task.get_logger().report_scalar("Best Model", "Loss", value=checkpoint_callback.best_model_score,
                                        iteration=trainer.current_epoch)

    # Завершение задачи ClearML
    task.close()


if __name__ == '__main__':
    main()
