import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


class Data:
    def __init__(self, images, rotations, transforms):
        self.images = images
        self.rotations = rotations
        self.transforms = transforms

    def ShowImage(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.images.shape[0])

        plt.imshow(self.images[idx])
        plt.show()

    def ShowRotation(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.rotations.shape[0])
        print(self.rotations[idx])

    def ShowTransform(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.transforms.shape[0])
        print(self.transforms[idx])
        


class DataLoader:
    def __init__(self, dataset_name):
        self.path_data = Path.cwd().joinpath("Data", dataset_name)

        self.train  = self.ReadDataFromJson("transforms_train.json")
        self.test   = self.ReadDataFromJson("transforms_test.json")
        self.val    = self.ReadDataFromJson("transforms_val.json")

    def ReadDataFromJson(self, fname):
        images = []
        rotations = []
        transforms = []

        try:
            data = json.load(open(self.path_data.joinpath(fname)))

            for frame in data["frames"]:
                imgPath = self.path_data.joinpath(f"{frame['file_path']}.png")
                images.append(ReadImage(imgPath))
                rotations.append(frame["rotation"])
                transforms.append(frame["transform_matrix"])

            images = np.array(images)
            rotations = np.array(rotations, np.float32)
            transforms = np.array(transforms, np.float32)

        except json.JSONDecodeError:
            print(f"{fname} is not a valid JSON file.")

        except Exception as e:
            print(e)

        print(f"Finished reading {fname}")

        return Data(images, rotations, transforms)
    
def ReadImage(path):
    try:
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)

    return image
