#type:ignore 
import os 
from pathlib import Path 
import cv2 
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CatsDogsDataset(Dataset): 
    
    def __init__(self, root, transform=None): 
        self.root = Path(root) 
        self.transform = transform 
        self.classes = os.listdir(self.root) 

        self.image_paths = [] 
        self.labels = [] 

        for class_name in self.classes: 
            class_path = self.root / class_name 
            for image_name in os.listdir(class_path): 
                image_path = class_path / image_name 
                self.image_paths.append(image_path)
                label = 0 if class_name == 'cat' else 1 
                self.labels.append(label) 
                # print(f"Loading image file: {image_path} with labels {label}")
                # break

    def __len__(self): 
        return len(self.labels) 
    
    def __getitem__(self, index): 
        image_path = self.image_paths[index] 
        label = self.labels[index] 

        image = cv2.imread(image_path) 
        # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform: 
            image = self.transform(image) 

        return image,label 


# if __name__ == "__main__":
#     # PROJECT_DIR = Path(__file__).resolve().parent.parent
#     # dataset = CatsDogsDataset(PROJECT_DIR / "archive" / "train") 
#     # image,label = dataset.__getitem__(1000) 
#     # cv2.imshow("Image", hsv_image)  
#     # # print(image.shape)
#     # # print(type(image))
#     # # print(image.dtype) 
#     # # plt.imshow(hsv_image)
#     # cv2.waitKey(0) 
#     # print(label)  
#     pass 