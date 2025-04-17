from lib import *
from make_datapath import make_datapath_list

class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        # Chua cac annotation cua anh
        ret = []

        # Read file xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # Chua thong tin cua boundingbox
            bndox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ["xmin", "ymin", "xmax", "ymax"]
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1

                if pt == "xmin" or pt == "xmax":
                    pixel = pixel / width       # ti le chieu rong
                else:
                    pixel = pixel / height      # ti le chieu cao

                bndox.append(pixel)
            
            label_id = self.classes.index(name) 
            bndox.append(label_id)          # Append class label

            ret += [bndox]
        
        return np.array(ret, dtype=np.float32)  # Chuyen ve dang numpy array
                                                # [[xmin, ymin, xmax, ymax, label_id], ......]


if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
    
    anno_xml = Anno_xml(classes)

    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)

    # lay ra anh so 1
    idx = 1
    img_file_path = val_img_list[idx]

    img = cv2.imread(img_file_path)  # Doc anh
    height, width, channels = img.shape  # Lay chieu cao va chieu rong cua anh

    # print("Image shape: ", img.shape)

    annotation_infor = anno_xml(val_annotation_list[idx], width, height)  # Lay annotation
    print(annotation_infor)