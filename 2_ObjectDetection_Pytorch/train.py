# 1. Dataloader
# 2. network -> SSD300
# 3. loss function -> MultiboxLoss
# 4. optimizer -> SGD (hoac Adam)
# 5. training, validation, test

import torch.backends
from lib import *
from make_datapath import make_datapath_list
from dataset import MyDataset, my_collate_fn
from data_transform import DataTransform
from extract_inform_annotation import Anno_xml
from model import SSD
from multiboxloss import MultiBoxLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True # de tang toc do train


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

net = SSD(phase="train", cfg=cfg)
vgg_weights = torch.load("./data/weights/vgg16_reducedfc.pth")
net.vgg.load_state_dict(vgg_weights) # Load weights from VGG16

def weight_init(m):     # m la model
    if isinstance(m, nn.Conv2d): # Neu la lop Conv2d
        nn.init.kaiming_normal_(m.weight.data) # Khoi tao weight bang phuong phap Kaiming
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0) # Khoi tao bias bang 0.0

# Kaiming He init
net.extras.apply(weight_init) # Khoi tao weight cho cac lop Conv2d trong net.extras
net.loc.apply(weight_init) # Khoi tao weight cho cac lop Conv2d trong net.loc
net.conf.apply(weight_init) # Khoi tao weight cho cac lop Conv2d trong net.conf

# print(net) # In ra model SSD

# Multibox Loss
# criterion = loss
criterion = MultiBoxLoss(jaccard_threshold=0.5, neg_pos=3, device=device) # Khoi tao loss function MultiboxLoss

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # Khoi tao optimizer SGD

# Training, validation
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=10):
    # move network to device (GPU or CPU)
    net.to(device)

    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []

    for epoch in range(num_epochs+1):
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print("---"*20)
        print(f"Epoch {epoch}/{num_epochs}")
        print("---"*20)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                print("Training...")
            else:
                if (epoch+1) % 10 == 0:
                    net.eval()
                    print("---"*10)
                    print("Validation...")
                else:
                    continue
            
            for images, targets in dataloaders_dict[phase]:
                # move data to device (GPU or CPU)
                images = images.to(device)
                targets = [anno.to(device) for anno in targets]

                # init optimizer
                optimizer.zero_grad()

                # forward 
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images) 

                    loss_l, loss_c = criterion(outputs, targets) # Tinh loss
                    loss = loss_l + loss_c
                        
                    if phase == "train":
                        loss.backward()         # Tinh gradient

                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0) # cat bo gradient

                        optimizer.step()        # Cap nhat weight

                        if iteration % 10 == 0:
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start
                            print(f"Iteration {iteration} - Loss: {loss.item():.4f} - 10iter: {duration:.2f}s")

                            # reset iteration time
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()  # Tinh loss tren tap train

                    else:
                        epoch_val_loss += loss.item()

        t_epoch_end = time.time()

        print("---"*20)
        print(f"Epoch {epoch} - Epoch train Loss: {epoch_train_loss:.4f} - Epoch Loss: {epoch_val_loss:.4f}")
        print(f"Duration: {t_epoch_end - t_epoch_start:.4f}s")

        t_epoch_start = time.time()

        log_epoch = {"epoch": epoch+1, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss}
        logs.append(log_epoch)

        df = pd.DataFrame(logs)
        df.to_csv("./data/ssd_log.csv")

        # Khoi tao lai loss cho epoch tiep theo
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Save model
        if ((epoch+1) % 10) == 0:
            torch.save(net.state_dict(), "./data/weights/ssd300_" + str(epoch+1) + ".pth")
            print(f"--Model saved at epoch {epoch+1}--")

num_epochs = 30
train_model(net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)






                         