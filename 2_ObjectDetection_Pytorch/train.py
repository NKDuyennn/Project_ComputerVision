# 1. Dataloader
# 2. network -> SSD300
# 3. loss function -> MultiboxLoss
# 4. optimizer -> Adam
# 5. training, validation, test

from lib import *
from make_datapath import make_datapath_list
from dataset import MyDataset, my_collate_fn
from data_transform import DataTransform
from extract_inform_annotation import Anno_xml
from model import SSD300


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# dataloader
root_path = "./data/VOCdevkit/VOC2012"
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(root_path)
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"]

color_mean = (104, 117, 123)  # Gia tri mean cua cac channel BGR
input_size = 300  # Kich thuoc input size

train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
val_dataset = MyDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

batch_size = 32
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

dataloader_dict = {
    "train": train_dataloader,
    "val": val_dataloader
}

# Network

cfg = {
    "num_classes": 21, # Number of classes (including background)
    "input_size": 300, # Input size for the SSD model
    "bbox_aspect_num": [4, 6, 6, 6, 4, 4], # Ti le khung hinh cho source1->source6
    "feature_maps": [38, 19, 10, 5, 3, 1], # Kich thuoc cua cac feature map tu source1->source6
    "steps": [8, 16, 32, 64, 100, 300], # Size of default box
    "min_size": [30, 60, 111, 162, 213, 264], # Kich thuoc cua default box
    "max_size": [60, 111, 162, 213, 264, 315], # Kich thuoc cua default box
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]], # Ti le khung hinh cho source1->source6
}

net = SSD300(phase="train", cfg=cfg)
