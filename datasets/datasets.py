import bisect
import copy
import torch
import torch.utils.data
from datasets.grouped_batch_sampler import GroupedBatchSampler

from datasets.mnist import MNIST, MNISTMetaData
from datasets.pascal import PascalVOC, PascalVOCMetaData
def compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def get_dataset(config, mode='train'):
    if config.dataset.startswith('mnist'):
        dataset = MNIST(config, mode)
        loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        meta_data = MNISTMetaData()
    elif config.dataset.startswith('VOC'):
        if mode=='valid':
            mode='val'
        # init dataset
        dataset = PascalVOC(config, mode)
        #init data sampler
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        # init batch sampler
        aspect_ratios = compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, [1])
        batch_sampler = GroupedBatchSampler(sampler, group_ids, config.batch_size, drop_uneven=False)
        # init batch collector
        collector = BatchCollector()
        # init data loader
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collector, num_workers=0, pin_memory=True)      
        meta_data = PascalVOCMetaData(config,mode)
    else:
        raise Exception('invalid dataset')
    
    return loader, meta_data

# modified based on 'maskrcnn-benchmark' 
# git@github.com:facebookresearch/maskrcnn-benchmark.git
class BatchCollector(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self):
        pass

    def __call__(self, batch):
        # transposed_batch = list(zip(*batch))
        # images = to_image_list(transposed_batch[0], self.size_divisible)
        # targets = transposed_batch[1]
        # img_ids = transposed_batch[2]
        max_h = max([sample[0].shape[1] for sample in batch]) 
        max_w = max([sample[0].shape[2] for sample in batch])
        image_tensor = torch.zeros([len(batch),batch[0][0].shape[0],max_h,max_w], device=torch.device('cpu'))
        keypoints_tensor = torch.zeros([len(batch),batch[0][1].shape[0],4], device=torch.device('cpu'))
        labels_tensor = torch.zeros([len(batch),batch[0][2].shape[0]], dtype=torch.long, device=torch.device('cpu'))

        for idx, sample in enumerate(batch):
            image, keypoints, labels = sample
            image_tensor[idx,:,:image.shape[1],:image.shape[2]] = image 
            keypoints_tensor[idx,:,:] = keypoints
            labels_tensor[idx,:] = labels

        return image_tensor, keypoints_tensor, labels_tensor
