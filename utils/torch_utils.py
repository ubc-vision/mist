import torch
import math

def to_gpu(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return tuple([t.cuda() for t in x])

def eval_accuracy (bboxes, bboxes_gt, labels, labels_gt, num_class, background_class=True):
    correct_detection = torch.zeros(labels.shape)
    # K [B, N, (x,y,w,h)]
    # labels [B, N]
    output = {}

    # total number of bbox in a batch
    N = (bboxes.shape[0] * bboxes.shape[1])
    
    # min IOU used for evaluation
    minIoU = 0.5

    # compute IoU between each keypoint and keypoint_gt
    bbox_min = (bboxes[:,:,0:2] - bboxes[:,:,2:4] / 2).unsqueeze(2) # [B, nk, 1, 2]
    bbox_max = (bboxes[:,:,0:2] + bboxes[:,:,2:4] / 2).unsqueeze(2)
    bbox_gt_min = (bboxes_gt[:,:,0:2] - bboxes_gt[:,:,2:4] / 2).unsqueeze(1) # [B, 1, nkg, 2]
    bbox_gt_max = (bboxes_gt[:,:,0:2] + bboxes_gt[:,:,2:4] / 2).unsqueeze(1)

    botleft = torch.max(bbox_min, bbox_gt_min)
    topright = torch.min(bbox_max, bbox_gt_max)

    inter = torch.prod(torch.nn.functional.relu(topright - botleft), dim=3)
    area_bbox = torch.prod(bbox_max - bbox_min, dim=3)
    area_bbox_gt = torch.prod(bbox_gt_max - bbox_gt_min, dim=3)
    union = area_bbox + area_bbox_gt - inter
    iou = inter / union # [B, k, kg, 1]   
    iou[iou != iou] = 0 

    # set iou of background class to 0
    if background_class:
        iou = (labels_gt!=0).unsqueeze(1).type(torch.float32)*(labels!=0).unsqueeze(-1).type(torch.float32) *iou     

    # total number of objects in batch
    if background_class:
        num_objects = (labels_gt!=0).sum().item() 
        num_detetcions = (labels!=0).sum().item() 
    else:
        num_objects = N
        num_detetcions = N         

    # generate label for visualization
    match_det = (torch.max(iou, dim=2)[0] > minIoU)
    selected_gt = torch.gather(labels_gt, dim=1, index=torch.max(iou, dim=2)[1])
    match_class = torch.eq(labels, selected_gt)
    output['keypoint_match_detection'] = torch.stack([match_det.float(), torch.max(iou, dim=2)[0],match_class.float()], dim=2)

    # compute detection accuracy
    acc_iou = ((torch.max(iou, dim=1)[0] > minIoU)).sum().float() / num_objects
    output['acc_det'] = acc_iou

    # prepare iou matrix for computing detection and classification accuracy
    labels_match = torch.eq(labels.unsqueeze(2), labels_gt.unsqueeze(1)) # [B, k, kg]
    iou = iou * labels_match.float()


    # prepare distance matrix for computing detection and classification accuracy (dist>1 outside gt bbox)
    dist = torch.max(torch.abs(bboxes[:,:,0:2].unsqueeze(2) - bboxes_gt[:,:,0:2].unsqueeze(1))/(bboxes_gt[:,:,2:4].unsqueeze(1)/2),dim=-1)[0]

    # replace invalid value in dist matrix to 2 so it won't be chosen
    dist[dist!=dist] =2
    dist[dist ==float('inf')] = 2

    # if label do not match, replace to 2
    dist[~labels_match] = 2

    # replace dist to background class to 2 
    if background_class:
        dist[(labels_gt==0).unsqueeze(1).repeat([1,labels.shape[1],1])] = 2


    # compute precision and recall
    tp_iou = 0
    tp_center = 0
    for b in range(bboxes_gt.shape[0]):
        for k in range(bboxes_gt.shape[1]):
            val, idx = torch.max(iou[b,:,k], dim=0)
            if val >= minIoU:
                iou[b, idx, :] = 0.0
                tp_iou += 1

            val, idx = torch.min(dist[b,:,k], dim=0)
            if val <= 1:
                dist[b, idx, :] = 2
                tp_center += 1
                correct_detection[b,idx] = 1
    if num_detetcions ==0:
        precision_iou = 0
        precision_center= 0
    else:
        precision_iou = tp_iou / num_detetcions
        precision_center= tp_center / num_detetcions            
    recall_iou = tp_iou/ num_objects
    recall_center = tp_center/ num_objects
    if recall_iou == 0 and precision_iou == 0:
        f1_iou = 0
    else:
        f1_iou = 2 * precision_iou * recall_iou / (precision_iou+recall_iou)
    if recall_center ==0 and precision_iou ==0:
        f1_center = 0
    else:
        f1_center = 2 * precision_center * recall_center / (precision_center+recall_center)


    output['f1_iou'] = f1_iou
    output['f1_center'] = f1_center
    output['precision_iou'] = precision_iou
    output['precision_center'] = precision_center
    output['recall_iou'] = recall_iou
    output['recall_center'] = recall_center   

    output['tp_center'] = tp_center
    output['tp_iou'] = tp_iou
    output['num_objects'] = num_objects
    output['num_detetcions'] = num_detetcions

    output['correct_detection'] = correct_detection

    # compute pure classification accuracy
    dt_1h = torch.nn.functional.one_hot(labels, num_class).sum(dim=1)[:,1:] 
    gt_1h = torch.nn.functional.one_hot(labels_gt, num_class).sum(dim=1)[:,1:]  
    acc_class_all_detect = 1.0 - torch.relu(gt_1h-dt_1h).sum().float()/num_objects

    output['acc_class'] = acc_class_all_detect

    return output


def gaussian(size, std=0.5):
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=size[0]), torch.linspace(0, 1, steps=size[1])])
    x = 2 * (x - 0.5)
    y = 2 * (y - 0.5)
    g = (x * x + y * y) / (2 * std * std)
    g = torch.exp(-g)
    g = g / (std * math.sqrt(2 * math.pi))
    return g

def gaussian2(size, center=None, std=0.5):
    if center is None:
        center = torch.tensor([[0.5, 0.5]])
    
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=size[0]), torch.linspace(0, 1, steps=size[1])])
    # print(x.unsqueeze(0).shape, .shape)
    x = 2 * (x.unsqueeze(0) - center[:,0,None,None])
    y = 2 * (y.unsqueeze(0) - center[:,1,None,None])
    g = (x * x + y * y) / (2 * std * std)
    g = torch.exp(-g)
    return g


def circle_mask(size, center=None, radius=0.5):
    if center is None:
        center = torch.tensor([[0.5, 0.5]])
    
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=size[0]), torch.linspace(0, 1, steps=size[1])])
    # print(x.unsqueeze(0).shape, .shape)
    x = 2 * (x.unsqueeze(0) - center[:,0,None,None])
    y = 2 * (y.unsqueeze(0) - center[:,1,None,None])
    d = (x * x + y * y) < (radius * radius)
    return d.float()

def half_mask(shape):
    angle = torch.rand((shape[0], 1, 1)) * math.pi * 2
    y, x = torch.meshgrid([torch.linspace(-1, 1, steps=shape[-2]), torch.linspace(-1, 1, steps=shape[-1])])
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    nx = torch.cos(angle)
    d = x * torch.cos(angle) + y * torch.sin(angle)
    mask = (d > 0).float()
    return mask

def square_mask(shape, size = 0.5):
    y, x = torch.meshgrid([torch.linspace(-1, 1, steps=shape[-2]), torch.linspace(-1, 1, steps=shape[-1])])
    d = torch.max(torch.abs(x), torch.abs(y))
    mask = (d < size).float()
    return mask

def calc_center_of_mass(heatmap,kernel_size):
    heatmap_exp = torch.exp(heatmap)
    heatmap_unf = torch.nn.functional.unfold(heatmap_exp, (kernel_size, kernel_size),padding = kernel_size//2).transpose(1,2)
    w_x = (torch.arange(kernel_size)-kernel_size//2).unsqueeze(0).expand(kernel_size,-1).reshape(-1,1).float()
    w_y = (torch.arange(kernel_size)-kernel_size//2).unsqueeze(1).expand(-1,kernel_size).reshape(-1,1).float()
    w_s = torch.ones(kernel_size,kernel_size).reshape(-1,1).float()
    heatmap_unf_x = heatmap_unf.matmul(w_x)
    heatmap_unf_y = heatmap_unf.matmul(w_y)
    heatmap_unf_s = heatmap_unf.matmul(w_s)
    offset_unf = torch.cat([heatmap_unf_x/heatmap_unf_s, heatmap_unf_y/heatmap_unf_s],dim=-1).transpose(1, 2)
    offset = torch.nn.functional.fold(offset_unf, (heatmap.shape[2], heatmap.shape[3]), (1, 1))
    grid_x = torch.arange(heatmap.shape[3]).unsqueeze(0).expand(heatmap.shape[2],-1).float()
    grid_y = torch.arange(heatmap.shape[2]).unsqueeze(1).expand(-1,heatmap.shape[3]).float()
    grid_xy = torch.cat([grid_x.unsqueeze(0),grid_y.unsqueeze(0)],dim=0)
    center = grid_xy+offset
    return center
def inverse_heatmap(keypoints, out_shape):
    heatmap = torch.zeros(out_shape)
    batch = torch.arange(keypoints.shape[0]).repeat(keypoints.shape[1], 1).permute(1, 0)
    x = keypoints[:, :, 0]
    y = keypoints[:, :, 1]
    x = torch.clamp((x + 0.5).long(), 0, out_shape[-1] - 1)
    y = torch.clamp((y + 0.5).long(), 0, out_shape[-2] - 1)
    heatmap[batch, 0, y, x] = 1.0
    return heatmap.detach()

def inverse_heatmap_gaussian(bboxes, out_shape, var_scale=0.125):
    # out_shape (B, 1, H, W)
    B, _, H, W = out_shape
    bboxes = torch.clamp(bboxes, 0.000001)
    index_x = torch.arange(W).repeat(H)  
    index_y = torch.arange(H).unsqueeze(-1).repeat(1,W).reshape(-1) 
    index = torch.cat((index_x.unsqueeze(-1),index_y.unsqueeze(-1)),dim=-1).float()
    exp_term = torch.matmul(torch.pow(((bboxes[:,:,:2].unsqueeze(-2)-index)/(bboxes[:,:,2:]*var_scale).unsqueeze(-2)),2),torch.tensor([[0.5],[0.5]])).squeeze()
    norm = torch.exp(-exp_term)#/(bboxes[:,:,[2]]*var_scale*bboxes[:,:,[3]]*var_scale)/2/math.pi
    heatmap = torch.sum(norm,dim=1).reshape(out_shape)

    return heatmap.detach()

def construct_dist_mat(kp_1, kp_2):
    # distance square matrix between two sets of points 
    # kp_1, kp_2 [B,N,2]
    xy_1_sq_sum_vec = torch.matmul(kp_1**2,torch.ones(2,1)) 
    xy_2_sq_sum_vec = torch.matmul(kp_2**2,torch.ones(2,1))
    # row: kp_1 column: kp_2
    xy_12_sq_sum_mat =  xy_1_sq_sum_vec + xy_2_sq_sum_vec.transpose(-1,-2)
    xy_mat = torch.matmul(kp_1, kp_2.transpose(-1,-2))
    dist_mat = xy_12_sq_sum_mat - 2*xy_mat
    dist_mat = torch.max(dist_mat,torch.zeros_like(dist_mat))  
    return dist_mat

def xys_to_xywh(boxes):
    return torch.cat([boxes, boxes[...,2,None]], dim=-1)

def xyxy_to_xywh(boxes):
    wh = (boxes[:,:,2:4] - boxes[:,:,0:2])
    center = (boxes[:,:,0:2] +wh/ 2)
    return torch.cat([center, wh], dim=2)

def xywh_to_xyxy(boxes):
    K_min = (boxes[:,:,0:2] - boxes[:,:,2:4] / 2)
    K_max = (boxes[:,:,0:2] + boxes[:,:,2:4] / 2)
    return torch.cat([K_min, K_max], dim=2)

def xys_to_xyxy(boxes):
    return xywh_to_xyxy(xys_to_xywh(boxes))

def scale_keypoints(boxes, scale=1.0):
    # boxes [B, N, 3/4] (xys or xywh)
    return torch.cat((boxes[:,:,:2], boxes[:,:,2:] * scale), dim=2)

if __name__ == '__main__':
    eps = 1e-5
    # test center of mass
    print("Testing clac_center_of_mass ...")
    heatmap = heatmap = torch.randn(32, 1, 5, 5)
    kernel_size = 3
    center = calc_center_of_mass(heatmap, kernel_size)
    # check shape
    if heatmap.shape[0] != center.shape[0] or heatmap.shape[2] != center.shape[2] or heatmap.shape[3] != center.shape[3]:
        raise Exception("output shape of calc_center_of_mass is different from input shape")
    # check calculation
    heatmap_exp = torch.exp(heatmap)
    c_x = 1+(-torch.sum(heatmap_exp[0,0,0:3,0])+torch.sum(heatmap_exp[0,0,0:3,2]))/torch.sum(heatmap_exp[0,0,0:3,0:3])
    c_y = 1+(-torch.sum(heatmap_exp[0,0,0,0:3])+torch.sum(heatmap_exp[0,0,2,0:3]))/torch.sum(heatmap_exp[0,0,0:3,0:3])
    if torch.abs(c_x - center[0,0,1,1])>eps or torch.abs(c_y - center[0,1,1,1])>eps:
        raise Exception("calc_center_of_mass output wrong result")
    print("Pass")

