from lib import *
from data_transform import DataTransform
from make_datapath import make_datapath_list
from extract_inform_annotation import Anno_xml

class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform  # Chuyen doi du lieu
        self.anno_xml = anno_xml # Chuyen doi annotation

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)  # Lay ra item tu index

        return img, gt  # Tra ve anh, annotation
    
    def pull_item(self, index):
        img_file_path = self.img_list[index]  # Lay ra duong dan den anh
        img = cv2.imread(img_file_path)
        height, width, channels = img.shape

        # Lay annotation
        anno_file_path = self.anno_list[index]  # Lay ra duong dan den annotation
        anno_info = self.anno_xml(anno_file_path, width, height)

        # Preprocess annotation
        img, boxes, labels = self.transform(img, self.phase, anno_info[:, :4], anno_info[:, 4])  # Chuyen doi du lieu

        # BGR -> RGB, [height, width, channels] -> [channels, height, width]
        torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # ground truth
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))  # Chuyen doi annotation ve dang [x1, y1, x2, y2, label]

        return img, gt, height, width  # Tra ve anh, annotation, height, width
    

if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
    
    # Chuan bi train, val, annotation list
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # Chuan bi chuyen doi du lieu
    color_mean = (104, 117, 123)  # Gia tri mean cua cac channel BGR
    input_size = 300  # Kich thuoc input size
    
    train_dataset = MyDataset(train_img_list, train_annotation_list, phase="train", 
                              transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
    val_dataset = MyDataset(val_img_list, val_annotation_list, phase="val",
                            transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
    
    # print(train_dataset.__len__())  # In ra so luong anh trong tap train
    print(train_dataset.__getitem__(1)) # In ra item dau tien trong tap train