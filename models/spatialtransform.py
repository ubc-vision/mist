import torch

def spatial_transformer(images, boxes, out_shape):
    # images: [B, C, H, W]
    # boxes:  [B, N, (x,y,w,h)]
    # out_shape: (h, w)
    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()

    B, C, H, W = images.shape
    N = boxes.shape[1]
    
    grid = meshgrid(out_shape).to(images.device).contiguous()
    grid = grid - 0.5
    grid = grid.repeat([B, N, 1, 1])  # shape: (B, N, 3, h*w)
    bx = boxes[:,:,0]
    by = boxes[:,:,1]
    bw = boxes[:,:,2]
    bh = boxes[:,:,3]
    
    x = (grid[:,:,:,0] * bw.unsqueeze(2) + bx.unsqueeze(2)).view(-1)
    y = (grid[:,:,:,1] * bh.unsqueeze(2) + by.unsqueeze(2)).view(-1)


    # x = (x / (W-1) - 0.5) * 2
    # y = (y / (H-1) - 0.5) * 2
    # wgrid = torch.stack([x, y], dim=-1).view(B*N, *out_shape, 2)
    # wimages = torch.repeat_interleave(images, N, dim=0)
    # output = grid_sample(wimages, wgrid, mode='bilinear')
    # output = output.view(B, N, C, *out_shape)
    # return output


    x0 = x.long()
    y0 = y.long()
    x1 = x0 + 1
    y1 = y0 + 1

    # clamp
    x = torch.clamp(x, 0, W - 1)
    y = torch.clamp(y, 0, H - 1)
    x0 = torch.clamp(x0, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # convert to linear indices
    batch_inds = torch.arange(B, device=images.device)
    
    batch_inds = torch.repeat_interleave(batch_inds, N)
    base = torch.repeat_interleave(batch_inds, out_shape[0]*out_shape[1], 0) * H * W
    
    idx_a = base + y0 * W + x0
    idx_b = base + y1 * W + x0
    idx_c = base + y0 * W + x1
    idx_d = base + y1 * W + x1

    # gather pixel values
    images = images.permute(0, 2, 3, 1).contiguous().view(-1, C)

    Ia = images[idx_a, :]
    Ib = images[idx_b, :]
    Ic = images[idx_c, :]
    Id = images[idx_d, :]

    # bilinear interpolation
    wa = ((x1.float() - x) * (y1.float() - y)).unsqueeze(1)
    wb = ((x1.float() - x) * (y - y0.float())).unsqueeze(1)
    wc = ((x - x0.float()) * (y1.float() - y)).unsqueeze(1)
    wd = ((x - x0.float()) * (y - y0.float())).unsqueeze(1)

    output = wa * Ia + wb * Ib + wc * Ic + wd * Id
    output = output.view(B, N, out_shape[0], out_shape[1], C).permute(0, 1, 4, 2, 3)
    return output

def patches_to_image(patches, keypoints, out_shape):
    # patches [B, N, C, H, W]
    # boxes:  [B, N, 4]
    # out_shape: (h, w)
    # returns -> [B, C, H, W]
    pass

def meshgrid(out_shape):
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=out_shape[-2]), torch.linspace(0, 1, steps=out_shape[-1])])
    x, y = x.flatten(), y.flatten()
    grid = torch.stack([x, y, torch.ones_like(x)], dim=1)
    return grid




if __name__ == '__main__':
    from scipy.misc import face
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import numpy as np

    np_image = imread('dog.jpg')
    images = torch.from_numpy(np_image)[None, ...].type(torch.FloatTensor)
    images = images.permute(0, 3, 1, 2)
    
    print(images.shape)
    boxes = torch.tensor([[[-100, -100, 1000, 700], [512, 340, 100, 100]]], dtype=torch.float32)

    patches = spatial_transformer(images, boxes, out_shape=[300, 300])
    patches = patches.permute(0, 1, 3, 4, 2)

    np_image = patches[0,1].numpy().astype(np.uint8)
    plt.imshow(np_image)
    plt.show()

    