from lib import *

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
class DefBox():
    def __init__(self, cfg):
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg["steps"]

    def create_defbox(self):
        defbox_list = []

        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size / self.steps[k] # Kich thuoc cua feature map

                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k

                # Small square box
                s_k = self.min_size[k]/self.img_size # Kich thuoc cua default box
                defbox_list += [cx, cy, s_k, s_k] # w=s_k, h=s_k
                # Big square box
                s_k_ = sqrt(s_k*(self.max_size[k]/self.img_size))
                defbox_list += [cx, cy, s_k_, s_k_]

                # Aspect ratio box
                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    defbox_list += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        output = torch.Tensor(defbox_list).view(-1, 4) # Chuyen ve tensor 2 chieu
        output.clamp_(max=1, min=0) # Chuan hoa ve [0,1]

        return output
    
if __name__ == "__main__":
    defbox = DefBox(cfg)
    dbox_list = defbox.create_defbox()
    # print(dbox_list.size()) # (8732, 4)
    # print(dbox_list) # In ra cac default box

    print(pd.DataFrame(dbox_list.numpy(), columns=["cx", "cy", "w", "h"]))