import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import v2 as T
from torch_utils import utils
from torch_classes.TorchCustomDatasets import CustomVOCDatasetTorchTuning
from torch_utils.engine import train_one_epoch, evaluate


# Классы для которых производится обучение модели
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


def get_model_object_detection(num_classes):
    # Загрузка весов предобученной модели
    model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")

    model.head.classification_head.num_classes = num_classes

    return model


# Преобразования данных в датасете
def get_transform(train):
    transforms = []
    if train:
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
dataset_train = CustomVOCDatasetTorchTuning(root="../Datasets/Foreign Objects v1", image_set="train",
                                            voc_classes=VOC_CLASSES, transforms=get_transform(False))
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=4,
    shuffle=True,
    collate_fn=utils.collate_fn
)

dataset_valid = CustomVOCDatasetTorchTuning(root="../Datasets/Foreign Objects v1", image_set="valid",
                                            voc_classes=VOC_CLASSES, transforms=get_transform(False))
valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=2,
    shuffle=False,
    collate_fn=utils.collate_fn
)

# Пример обучения модели
# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images, targets)  # Returns losses and detections
# print(output)
#
# # Инференс модели
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)  # Returns predictions
# print(predictions[0])

# Обучение на GPU если доступно
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Определение кол-ва классов для дообучения модели
num_classes = 20

# Определение кастомной модели
model = get_model_object_detection(num_classes)

# Загрузка модели на CPU или GPU
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# Обучение модели
num_epochs = 50

for epoch in range(num_epochs):
    # Обучение 1 эпоха с выводом loss каждые 10 итераций
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # Обновление скорости обучения
    lr_scheduler.step()
    # Валидация обучения модели на валидационных данных
    evaluate(model, valid_loader, device=device)

print("Обучение завершено!")
