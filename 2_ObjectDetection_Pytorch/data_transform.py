from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans
from make_datapath import make_datapath_list
from extract_inform_annotation import Anno_xml
from lib import *

class DataTransform():
    def __init__(self, input_size, color_mean):

        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),      # Convert all images to float
                ToAbsoluteCoords(),     # Tra ve annotation dang ban dau cua anh
                PhotometricDistort(),   # Thay doi mau sac dung ham random
                Expand(color_mean),     # Mo rong anh  
                RandomSampleCrop(),     # Lay ngau nhien mot phan cua anh
                RandomMirror(),         # Lay anh doi xung
                ToPercentCoords(),      # Chuyen doi toa do ve 0-1
                Resize(input_size),     # Resize anh ve kich thuoc input_size
                SubtractMeans(color_mean) # Tru di gia tri mean 
            ]),
            "val": Compose([
                ConvertFromInts(),      # Convert all images to float
                Resize(input_size),     # Resize anh ve kich thuoc input_size
                SubtractMeans(color_mean) # Tru di gia tri mean 
            ])
        }
    
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)

if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
    
    # Chuan bi train, val, annotation list
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # Doc anh 
    # Lay ra anh so 1
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)  # Height, Width, Channels(BGR)
    height, width, channels = img.shape

    # Annotation information
    trans_anno = Anno_xml(classes)
    anno_info_list = trans_anno(train_annotation_list[0], width, height)  # Lay annotation

    # plot orginal image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    # Mac dinh cv2 doc anh theo BGR, chuyen sang RGB de ve
    plt.show()

    # Chuan bi chuyen doi du lieu
    color_mean = (104, 117, 123)  # Gia tri mean cua cac channel BGR
    input_size = 300  # Kich thuoc input size
    transform = DataTransform(input_size, color_mean)  # Khoi tao doi tuong chuyen doi

    # Tranform train img
    phase = "train"  # Chon phase train
    img_transformed, boxes_transformed, labels_transformed = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])  # Chuyen doi du lieu
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))    # Mac dinh cv2 doc anh theo BGR, chuyen sang RGB de ve
    plt.show()

    # Transform val img
    phase = "val"  # Chon phase val
    img_transformed, boxes_transformed, labels_transformed = transform(img, phase, anno_info_list[:, :4], anno_info_list[:, 4])  # Chuyen doi du lieu
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))    # Mac dinh cv2 doc anh theo BGR, chuyen sang RGB de ve
    plt.show()