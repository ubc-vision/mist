import os
import torch
import torch.utils.data
import numpy as np
import skimage.io
import skimage.transform

MNIST_CLASSES = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
)

class MNISTMetaData():
    def __init__(self):
        self.cls = MNIST_CLASSES
    def get_num_class(self):
        return len(self.cls)
    def get_class_name(self, class_id):
        return self.cls[class_id]

class MNIST(torch.utils.data.Dataset):
    def __init__(self, config, mode):
        self.root_dir = config.dataset_dir+'/'+config.dataset+'/'
        self.image_paths = np.genfromtxt(self.root_dir + mode + '.txt', delimiter=',', dtype='str', encoding='utf-8')
        self.labels = np.genfromtxt(self.root_dir + mode +'_labels.txt', delimiter=',', dtype='int', encoding='utf-8')
        self.keypoints = np.load(self.root_dir + mode +'_keypoints.npy')
        self.num_kp = config.k
        self.image_size =config.image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = skimage.io.imread(img_name)
        image = skimage.transform.resize(image,(self.image_size,self.image_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.clamp(image, 0.0, 1.0)

        # load keypoints
        keypoints = torch.from_numpy(self.keypoints[idx].copy())
        keypoints[:,2] = keypoints[:,2] * 2.0
        keypoints = torch.cat((keypoints,keypoints[:,[2]]), axis=-1)
        stride = self.image_size/image.shape[1]
        keypoints = keypoints*stride

        # load label
        labels = torch.from_numpy(self.labels[idx])

        return image, keypoints, labels