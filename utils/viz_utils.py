import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def save_bbox_images(image, bbox, label, name, path, background_class):
    image = image.cpu().permute(1, 2, 0).numpy()
    image = draw_boxes_on_images(image, bbox, label, background_class)
    image.save(os.path.join(path,'{}.jpg'.format(name)),quality=95)

def draw_boxes_on_images(image, bbox, label, background_class):
    scale = 224/max(image.shape)
    H = int(scale*image.shape[0])
    W = int(scale*image.shape[1])
    bbox = bbox*scale
    fnt = ImageFont.load_default()
    image = Image.fromarray(np.uint8(image*255), 'RGB')
    image = image.resize((W, H))
    draw = ImageDraw.Draw(image)
    for i in range(bbox.shape[0]):
        if background_class and label[i]=='__background__':
            continue
        kp = bbox[i,:].tolist()
        color = (0,0, 255)
        draw.rectangle(kp, outline=color, fill=None)
        color = (0,255,0)
        if kp[0]<0:
            x_0 = kp[2] - len(label[i])*8
        else:
            x_0 = kp[0] + 1
        if kp[1]<0:
            x_1 = kp[3] - 12
        else:
            x_1 = kp[1] + 1
        draw.text((x_0, x_1), label[i], width=12, fill=color, font=fnt)
    return image