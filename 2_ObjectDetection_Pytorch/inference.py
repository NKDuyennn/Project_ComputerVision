from lib import *
from model import SSD
from data_transform import DataTransform

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"]
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

net = SSD(phase="test", cfg=cfg)
net_weights = torch.load("./data/weights/ssd300_mAP_77.43_v2.pth", map_location={"cuda:0": "cpu"})
net.load_state_dict(net_weights) 

def show_predict(img_file_path):
    img = cv2.imread(img_file_path) # Doc hinh tu file

    color_mean = (104, 117, 123)  # Gia tri mean cua cac channel BGR
    input_size = 300  # Kich thuoc input size

    transform = DataTransform(input_size, color_mean) # Chuyen doi hinh anh

    phase = "val"   # Chuyen doi hinh anh cho phase "val"
    img_transformed, boxes, labels = transform(img, phase, "", "") # Chuyen doi hinh anh
    img_tensor = torch.from_numpy(img_transformed[:,:,(2,1,0)].permute(2,0,1)) # Chuyen doi hinh anh sang tensor

    net.eval() # Chuyen sang che do test
    input = img_tensor.unsqueeze(0) # (1, 3, 300, 300)
    output = net(input) # Dua hinh vao model

    plt.figure(figsize=(10, 10))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    font = cv2.FONT_HERSHEY_SIMPLEX

    detections = output.data # Lay ket qua du doan (1, 21, 200, 5) 5: score, cx, cy, w, h
    scale = torch.Tensor([img.shape[1::-1]]).repeat(2) # Chuyen doi scale

    for i in range(detections.size(1)): # Duyet qua cac class
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0].item() # Lay score
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy() # Lay vi tri cua khung hinh
            cv2.retangle(img,
                        (int(pt[0]), int(pt[1])),
                        (int(pt[2]), int(pt[3])),
                        colors[i%3], 2
            )

            display_text = f"{classes[i]}: {score:.2f}"

            cv2.putText(img, display_text, (int(pt[0]), int(pt[1])),
                        font, 0.5, colors[i%3], 1, cv2.LINE_AA)
            
            j += 1
        
    cv2.imshow("Detection Result", img) # Hien thi hinh anh
    cv2.waitKey(0) # Cho den khi nhan phim
    cv2.destroyAllWindows() # Dong cua so

if __name__ == "__main__":
    img_file_path = "./data/test.jpg" # Duong dan den hinh anh
    show_predict(img_file_path) # Goi ham show_predict de hien thi ket qua du doan