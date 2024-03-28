# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lim
"""
import os
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
from backbone.resnet import ResNet
from backbone.dlanet import DlaNet
from Loss import _gather_feat
from shutil import copy, rmtree
from PIL import Image, ImageDraw
from dataset import get_affine_transform
from Loss import _transpose_and_gather_feat
from PIL import Image
from torchvision import transforms

def mk_file(file_path: str):
    if os.path.exists(file_path):
        return
    os.makedirs(file_path)

def rotateImage(img,degree,x,y,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
    matRotation=cv2.getRotationMatrix2D((x,y),degree,1)
    # matRotation[0, 2] += (widthNew - width) / 2
    # matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (2000,2000), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgRotation = imgRotation[int(pt3[0]):int(pt1[0]),int(pt1[1]):int(pt3[1])]
    return imgRotation


def rotate_point(point, center, angle_rad):
    """Rotate a point around a center point by a given angle."""
    x, y = point
    cx, cy = center
    new_x = (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad) + cx
    new_y = (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad) + cy
    return new_x, new_y

def draw(filepath1, filepath2, filename):
    # 读取图像
    image = Image.open(filepath1 + filepath2 + '/' + filename)
    if not res:
        print("cropping and saving fail!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: " + filename.split('.')[0] + '.png')
        image.save(os.path.join(path_name_all+'/fail_images', filename.split('.')[0] + '.png'))
    for class_name,lx,ly,rx,ry,ang, prob in res:
        result = [int((rx+lx)/2),int((ry+ly)/2),int(rx-lx),int(ry-ly),ang]
        result=np.array(result)
        x=int(result[0])
        y=int(result[1])
        height=int(result[2])
        width=int(result[3])
        angle = result[4]

        # 指定新的尺寸
        lenge = max(image.width, image.height)
        new_width = lenge * 3
        new_height = lenge * 3

        # 计算左上角的坐标，以在新尺寸内居中显示图像
        left = (new_width - image.width) // 2
        top = (new_height - image.height) // 2

        # 创建新的画布，填充为黑色
        padded_image = Image.new('RGB', (new_width, new_height), color='black')

        # 将原始图像粘贴到新的画布中
        padded_image.paste(image, (left, top))
        center_x = x + (new_width - image.width) // 2
        center_y = y + (new_height - image.height) // 2
        # 将角度转换为弧度
        angle_rad = math.radians(angle)

        # 计算旋转框的四个角点坐标
        top_left = (center_x - width / 2, center_y - height / 2)
        top_right = (center_x + width / 2, center_y - height / 2)
        bottom_left = (center_x - width / 2, center_y + height / 2)
        bottom_right = (center_x + width / 2, center_y + height / 2)

        # 得到新的旋转框的四个顶点坐标
        new_rect_points = [top_left, top_right, bottom_right, bottom_left]

        rotated_image = padded_image.rotate(-angle + 90, center=(center_x, center_y), resample=Image.BICUBIC, expand=False)

        # 得到新的旋转框的坐标范围
        min_x = min(point[0] for point in new_rect_points)
        max_x = max(point[0] for point in new_rect_points)
        min_y = min(point[1] for point in new_rect_points)
        max_y = max(point[1] for point in new_rect_points)

        # 裁剪旋转后的图像
        cropped_image = rotated_image.crop((min_x, min_y, max_x, max_y))

        # 保存结果
        images_save_path = os.path.join(path_name_all, filepath2)
        cropped_image.save(os.path.join(images_save_path, filename.split('.')[0] + '.png'))
        print("cropping and saving successfully: " + filename.split('.')[0] + '.png')
        break

def pre_process(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([width / 2.,  height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height),flags=cv2.INTER_LINEAR)

    mean = np.array([0.5194416012442385,0.5378052387430711,0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)
    
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width) # 三维reshape到4维，（1，3，512，512） 
    
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // 4, 
            'out_width': inp_width // 4}
    return images, meta


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float() 
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, ang, reg=None, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
   
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    
    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, 1)

    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2,
                        ang], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def process(images, return_time=False):
    with torch.no_grad():
      output = model(images)
      hm = output['hm'].sigmoid_()
      ang = output['ang'].relu_()
      wh = output['wh']
      reg = output['reg'] 
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, ang, reg=reg, K=100)
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def post_process(dets, meta):  
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  
    num_classes = 1
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
      dets[0][j][:, :5] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
      kth = len(scores) - max_obj_per_img
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, 2 + 1):
        keep_inds = (detections[j][:, 5] >= thresh)
        detections[j] = detections[j][keep_inds]
    return detections



if __name__ == '__main__':
    model = ResNet(34)
    # model = DlaNet(34)
    device = torch.device('cuda')
    pth_name='all_res34_'
    path_name_all = pth_name + 'img_output_all'
    model.load_state_dict(torch.load(pth_name + 'best.pth'))
    model.eval()
    model.cuda()
    mk_file(path_name_all+'/fail_images/')
    images_path_all = [
        "CASIA",
        "XJTU-UP",
        "MPD",
        "IITD",
        "BJTU",
    ]
    img_path = '/home/lee/data2/data_base/'

    for path in images_path_all:

        images_save_path = os.path.join(path_name_all, path)
        mk_file(images_save_path)
        for image_name in os.listdir(img_path+path):
            original_img = Image.open(img_path + path + '/' + image_name)


            data_transform = transforms.Compose([transforms.ToTensor()])
            if len(original_img.split()) < 3:
                img1 = cv2.imread(img_path + path + '/' + image_name)
                gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2 = np.zeros_like(img1)
                img2[:, :, 0] = gray
                img2[:, :, 1] = gray
                img2[:, :, 2] = gray
                img = img2
            elif len(original_img.split()) > 3:
                img1 = cv2.imread(img_path + path + '/' + image_name)
                img2 = img1.convert('RGB')
                img = img2
            else:
                img = cv2.imread(img_path + path + '/' + image_name)

            images, meta = pre_process(img)
            images = images.to(device)
            output, dets, forward_time = process(images, return_time=True)

            dets = post_process(dets, meta)
            ret = merge_outputs(dets)

            res = np.empty([1,7])
            for i, c in ret.items():
                tmp_s = ret[i][ret[i][:,5]>0.3]
                tmp_c = np.ones(len(tmp_s)) * (i+1)
                tmp = np.c_[tmp_c,tmp_s]
                res = np.append(res,tmp,axis=0)
            res = np.delete(res, 0, 0)
            res = res.tolist()
            draw(img_path, path, image_name)
