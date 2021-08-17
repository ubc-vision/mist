import torch
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO

def figure_to_numpy(figure, close=True):
    buf = BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', transparent=False, dpi=300)
    if close:
        plt.close(figure)
    buf.seek(0)
    arr = matplotlib.image.imread(buf, format='png')[:,:,:3]
    arr = np.moveaxis(arr, source=2, destination=0)
    return arr

class CustomSummaryWriter(SummaryWriter):
    # def __init__(self):
    #     super(CustomSummaryWriter, self).__init__(

    def add_images_heatmap(self, name, images, heatmap, iteration):
            heatmap = self.draw_heatmap_on_images(images, heatmap)
            self.add_images(name, heatmap, iteration)

    def add_images(self, name, images, iteration, boxes_infer=None, boxes_gt= None, labels=None, resize=None, match=None):
        # images [B, C, H, W]
        max_images = min(images.shape[0],20)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if len(images.shape) == 4:
            images = images[0:max_images, ...]
        elif len(images.shape) == 5:
            images = images[0:max_images, ...]
            images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            raise Exception('images.shape() {}'.format(images.shape))

        if resize is not None:
            w = int(images.shape[-1] * resize)
            h = int(images.shape[-2] * resize)
            images = torch.nn.functional.interpolate(images, (h,w), mode='nearest')

        if (images.shape[1]!=3):
            images = torch.mean(images,dim=1).unsqueeze(1).repeat(1,3,1,1)
        images = ((images - images.min()) / (images.max() - images.min()) * 255.0).byte()
        # images = (images * 255.0).byte()
        if boxes_infer is not None:
            if resize is not None:
                boxes_infer = boxes_infer * resize
            images = self.draw_boxes_on_images(images, boxes_infer, labels, match)

        if boxes_gt is not None:
            if resize is not None:
                boxes_gt = boxes_gt * resize
            images = self.draw_boxes_on_images(images, boxes_gt)

        image = torchvision.utils.make_grid(images, nrow=max_images, padding=1, pad_value=255)
        self.add_image(name, image, iteration)


    def draw_boxes_on_images(self, images, boxes, labels=None, match=None):
        fnt = ImageFont.load_default()
        image_np = np.zeros(images.shape, dtype=np.uint8)
        for i in range(images.shape[0]):
            img = images[i,...].cpu().permute(1, 2, 0).numpy()
            H,W,_ = img.shape
            img = Image.fromarray(img, 'RGB')
            draw = ImageDraw.Draw(img)
            for j in range(boxes.shape[1]):
                kp = boxes[i,j,:].tolist()
                if match is not None:
                    color = (0,255,0) if int(round(match[i,j,0].item())) else (255,0,0)
                    draw.rectangle(kp, outline=color, fill=None)
                    draw.text((kp[2]-20, kp[3]-20), '{}'.format(int(match[i,j,1].item()*100)), fill=color, font=fnt)
                    if labels is not None:
                        color = (0,255,0) if int(round(match[i,j,2].item())) else (255,0,0)
                        draw.text((kp[0]+2, kp[1]), str(labels[i][j].item()), fill=color, font=fnt)
                        draw.text((W-15*(j+1), H-15), str(labels[i][j].item()), fill=(100,100,0), font=fnt)
                else: 
                    color = (0,0, 255)
                    draw.rectangle(kp, outline=color, fill=None)


            img = np.asarray(img)
            image_np[i,...] = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(image_np)

    def draw_heatmap_on_images(self, images, heatmap):
        plt.switch_backend('agg')
        # plt.set_cmap('jet')
        plt.set_cmap('jet')
        hmin=heatmap.min()
        hmax=heatmap.max()
        fig, axes = plt.subplots(1, images.shape[0], figsize=(16,2))
        if images.shape[0] == 1:
            axes = np.array([axes])
        for i in range(images.shape[0]):
            I = images[i,...].cpu().permute(1, 2, 0).numpy()
            H = heatmap[i,...].detach().cpu().numpy()
            ax = axes[i]
            ax.imshow(I)
            im = ax.imshow(H, alpha=1.0, vmin=hmin, vmax=hmax)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.colorbar(im, ax=axes.ravel().tolist())
        output = figure_to_numpy(fig)

        return torch.from_numpy(output)
