import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2



class MyDataset(Dataset) :
    def __init__(self, path, transforms=None):
        types = ['*.jpg', '*.jpeg']
        self.path = []
        for type in types:
            expath = glob.glob(os.path.join(path, "*", type), recursive=True)
            self.path.extend(expath)
        self.transforms = transforms
        self.sub_folders = os.listdir(path)
        self.label_dict = {} #0~5
        self.label_dict_test = {} # #Label order 1 to 6
        for i, folder in enumerate(self.sub_folders) :
            self.label_dict[folder] = i
            self.label_dict_test[i] = folder
        # print(f"label >> {self.label_dict}")
        # print(f"label str test >> {self.label_dict_test}")


    def __getitem__(self, item):
        image_path = self.path[item]
        ## cv2로 읽기
        img_array = np.fromfile(image_path, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # rgb로 변경
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # PIL로 전환
        image = Image.fromarray(image)
        
        folder_name = image_path.split('\\')[-2]
        # print(f"folder_name = {folder_name}")
        label_number = self.label_dict[folder_name]
        # print(f"folder name: {folder_name}, label number:{label_number}")

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label_number

    def __len__(self):
        return len(self.path)

if __name__ == "__main__":
    test = MyDataset("Dataset path", transforms=None)
    print("hello")
    print(len(test))
    for i in test:
        print(i)

    exit()
