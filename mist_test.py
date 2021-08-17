import torch
from tqdm import tqdm
import os

from models.mist import MIST
from datasets.datasets import get_dataset
from utils.summary import CustomSummaryWriter
from utils.config_utils import print_config, str2bool, get_mist_config
from utils.viz_utils import save_bbox_images
from utils.torch_utils import to_gpu, eval_accuracy, xywh_to_xyxy

if __name__ == '__main__':
    # get config
    config = get_mist_config()
    print_config(config)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # init data loader
    config.rand_horiz_flip = False
    config.rand_maskout = False 
    config.batch_size = 1
    dataset_te, metadata_te = get_dataset(config, mode='test') 
    
    # init network
    config.pretrained_resnet = False
    mist = MIST(config)
    
    # load pretrained weights
    best_model_path = os.path.join(config.model_dir,config.name + '_best_models')
    mist.load_state_dict(torch.load(best_model_path))
    
    # set evaluation metric 
    if config.dataset.startswith('mnist'):
        background_class=False
        val_metric = 'iou'
    else:
        background_class=True
        val_metric = 'center'
               
    # create viz folder
    test_results_path = os.path.join(config.test_path, config.name)
    if not os.path.isdir(test_results_path):
        os.makedirs(test_results_path)    
        
    # inference
    output = {}
    output['tp_center'] = 0
    output['tp_iou'] = 0
    output['num_objects'] = 0
    output['num_detetcions'] = 0
    for i, data in tqdm(enumerate(dataset_te)):
        images, keypoints_gt, labels_gt = to_gpu(data)
        B = images.shape[0]
        # forward pass
        bboxs, labels, _ = mist.forward(images.clone())
        # evaluation
        eval_test = eval_accuracy(bboxs, keypoints_gt, labels, labels_gt, 
            config.num_classes, background_class)
        for j,(image, bbox, label) in enumerate(zip(images,xywh_to_xyxy(bboxs),labels)):
            save_bbox_images(image, bbox, 
                [metadata_te.get_class_name(_label) for _label in label],str(i*B+j),
                test_results_path, background_class)
        # accumulate results
        output['tp_center'] += eval_test['tp_center']
        output['tp_iou'] += eval_test['tp_iou']
        output['num_objects'] += eval_test['num_objects']
        output['num_detetcions'] += eval_test['num_detetcions']

    # calculate f1 score       
    if eval_test['num_detetcions'] == 0:
        precision = 0
    else:
        precision = output['tp_'+val_metric]/output['num_detetcions']
    recall = output['tp_'+val_metric]/output['num_objects']
    if (precision + recall) == 0:
        f1 = 0
    else: 
        f1 = 2*precision*recall/(precision+recall)
    print('test set f1 {} score: {}'.format(val_metric, f1))