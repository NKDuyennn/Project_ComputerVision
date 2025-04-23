from lib import *
from l2_norm import L2Norm
from default_box import DefBox

def create_vgg():
    layers=[]
    in_channels=3

    cfgs = [64, 64, 'M', 128, 128, 'M',
            256, 256, 256, 'MC', 512, 512, 512, 'M',
            512, 512, 512]
    
    for cfg in cfgs:
        if cfg == 'M':  # floor lam tron xuong
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'MC':    #ceiling lam tron len
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]   # inplace=True: khong tao ra tensor moi, ma chinh sua tensor cu
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Maxpooling 3x3 stride=1 padding=1
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  
    relu6 = nn.ReLU(inplace=True)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
    relu7 = nn.ReLU(inplace=True)

    layers += [pool5, conv6, relu6, conv7, relu7]  # Them cac lop vao layers

    return nn.ModuleList(layers)  # Tra ve moduleList chua cac lop CNN
    
def create_extras():
    layers =[]
    in_channels = 1024

    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3)]
    layers += [nn.ReLU(inplace=True)]

    return nn.ModuleList(layers)  # Tra ve moduleList chua cac lop CNN

def create_loc_conf(num_classes=21, bbox_ratio_num=[4, 6, 6, 6, 4, 4]):
    loc_layers=[]
    conf_layers=[]

    # source1
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]

    # source2
    # loc
    loc_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]

    # source3
    # loc
    loc_layers += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size=3, padding=1)]

    # source4
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]

    # source5
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[4]*num_classes, kernel_size=3, padding=1)]

    # source6
    # loc
    loc_layers += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
    # conf
    conf_layers += [nn.Conv2d(256, bbox_ratio_num[5]*num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)  # Tra ve moduleList chua cac lop CNN

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

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        self.input_size = cfg["input_size"]

        # create main modules
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(self.num_classes, cfg["bbox_aspect_num"])
        self.L2Norm = L2Norm()

        # create default boxes
        dbox = DefBox(cfg)
        self.dbox_list = dbox.create_defbox()

        if phase == "inference":
            self.detect = Detect()


def decode(loc, defbox_list):
    """
    parameters:
        loc: [8732, 4] (delta x, delta y, delta w, delta h)
        defbox_list: [8732, 4] (cx_def, cy_def, w_def, h_def)
    return:
        boxes: [xmin, ymin, xmax, ymax]
    """

    boxes = torch.cat((defbox_list[:, :2] + 0.1*loc[:, :2]*defbox_list[:, 2:], # cx, cy
                       defbox_list[:, 2:]*torch.exp(loc[:, 2:]*0.2)), dim=1) # w, h
    boxes[:, :2] -= boxes[:, 2:]/2  # xmin, ymin
    boxes[:, 2:] += boxes[:, :2]  # xmax, ymax

    return boxes.clamp_(min=0, max=1)  # Chuan hoa ve [0,1]

# non_max_suppression: Ham loai bo cac box trung lap
def nms(boxes, scores, overlap=0.45, top_k=200):
    """
    parameters:
        boxes: [Num_boxes, 4] (xmin, ymin, xmax, ymax)
        scores: [Num_boxes, num_classes]
        overlap: Ti le trung lap
        top_k: So luong box duoc chon
    return:
        keep: cac box duoc chon
    """
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()  # Danh sach cac box duoc chon

    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax

    # area = (x2 - x1 + 1) * (y2 - y1 + 1)  # Dien tich cua box
    area = torch.mul(x2-x1, y2-y1)  # Dien tich cua box

    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(0)  # Sap xep cac box theo score
    idx = idx[-top_k:]  # Lay top_k box co score cao nhat
    
    while idx.numel() > 0: # khi idx khong rong
        i = idx[-1]
        
        keep[count] = i  # Luu lai box co score cao nhat
        count += 1
        
        if idx.size(0) == 1:  # Neu chi con 1 box
            break

        idx = idx[:-1]  # Lay tat ca cac box khong phai i

        # information of boxes
        torch.index_select(x1, 0, idx, out=tmp_x1)  # Chon cac box tu x1
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # x1[i] if  tmp_x1 < x1[i] else tmp_x1
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i]) # x2[i] if tmp_y2 > y2[i] else tmp_y2

        # Chuyen ve tensor co size ma index duoc giam di 1
        tmp_w.resize_as(tmp_x2)  # Tao tensor tmp_w co kich thuoc nhu tmp_x2
        tmp_h.resize_as(tmp_y2)  # Tao tensor tmp_h co kich thuoc nhu tmp_y2
        
        tmp_w = tmp_x2 - tmp_x1  # Chieu rong cua box
        tmp_h = tmp_y2 - tmp_y1  # Chieu cao cua box

        tmp_w = torch.clamp(tmp_w, min=0.0)  # Chuan hoa ve [0,1]
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # Dien tich giao nhau
        inter = tmp_w * tmp_h
        # Dien tich cac box khac i 
        others_area = torch.index_select(area, 0, idx)
        # Dien tich hop
        union = area[i] + others_area - inter

        # Ti le giao/ hop
        iou = inter / union

        idx = idx[iou.le(overlap)]  # Chon cac box co iou nho hon overlap

    return keep, count  # Tra ve danh sach cac box duoc chon va so luong box duoc chon





if __name__ == "__main__":
    # vgg = create_vgg()
    # print(vgg)  # In ra cac lop CNN
    # extras = create_extras()
    # print(extras)  # In ra cac lop CNN

    # loc, conf = create_loc_conf()
    # print(loc)  # In ra cac lop CNN
    # print(conf)  # In ra cac lop CNN

    ssd = SSD("train", cfg)
    print(ssd)  # In ra cac lop CNN