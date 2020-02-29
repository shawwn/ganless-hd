#from pycocotools.coco import COCO
import numpy as np
#import skimage.io as io

#np.random.seed(1)

import torch
from PIL import Image, ImageDraw
import torch as t
from torchvision.transforms import ToPILImage
from torch.utils.data.dataset import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import scipy.misc
import cv2
import random
import os
import scipy
from scipy import ndimage
import math
import random
import torch.nn.functional as F

def tensor2img(img):
  if img.shape[0] == 3:
    img = img.permute(1,2,0)
  try:
    img = img.numpy()
  except:
    pass
  img = norm2img(img)
  return img

def img2scene(img):
  if isinstance(img, str):
    img = cv2.imread(img).copy()
  #img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).copy()
  #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
  return img

def scene2img(img):
  img = tensor2img(img)
  if False:
    img[:,:,1] = img[:,:,0]
    img[:,:,2] = img[:,:,0]
  else:
    #img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB).copy()
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    pass
  return img

def resize(*args, **kws):
  return cv2.resize(*args, **kws, interpolation = cv2.INTER_AREA)

def downsize(img, target):
  while img.shape[0]>>1 >= target[0] and img.shape[1]>>1 >= target[1]:
    img = resize(img, (img.shape[0] // 2, img.shape[1] // 2))
  img = resize(img, target)
  return img

def crop(img, x, y, w, h):
  ih, iw, ic = img.shape
  assert x >= 0
  assert y >= 0
  assert x <= iw - w
  assert y <= ih - h
  assert ic == 3
  print(iw, ih, w, h, x, y, iw - x, ih - y)
  #img = img[int(x):int(x+w), int(y):int(y+h)]
  img = img[int(y):int(y+h), int(x):int(x+w)]
  return img

def rand(a, b):
  if a > b:
    a, b = b, a
  if a == b:
    return a
  #return np.random.randint(int(a), int(b+1))
  return np.random.uniform(a, b)

def randomBox(iw, ih, w, h):
  s = ih / h
  w *= s
  h *= s
  if w > iw:
    s = iw / w
    w *= s
    h *= s
  y = rand(0.0, ih - h)
  x = rand(0.0, iw - w)
  return x, y, w, h

def randomCrop(img, w, h):
  ih, iw, ic = img.shape
  x, y, w, h = randomBox(iw, ih, w, h)
  img = crop(img, x, y, w, h)
  return img

def loadRandomImage(paths, *args, **kws):
  path = np.random.choice(paths)
  return loadImage(path, *args, **kws)

def img2norm(img):
  img = (((img.astype(np.float32) ** 1.0) / 255.0) * 2) - 1
  return img

def norm2img(img):
  img += 1
  img /= 2
  img *= 255
  return img

def img2tensor(img):
  # normalize
  #img =  (((img.astype(np.float32) ** 1.0) / 255.0) * 2) - 1
  img = img2norm(img)
  img = torch.from_numpy(img.astype(np.float32) )
  img = img.permute(2, 0, 1)
  return img

def downsample(img, size, target=None):
  if target is None:
    target = (img.shape[1], img.shape[0])
  img = resize(resize(img, size), target)
  return img

def edgedetect(dst, img):
  dst = dst.copy()
  img = img.copy()
  #img = resize(img, (dst.shape[1], dst.shape[0]))
  for i in range(3):
    img[:,:,i] = cv2.Canny(img[:,:,i], 1, 255)
  idx = img == 0	
  img[idx] = dst[idx]
  return img

def loadImage(path, img_size, shrink_size):
  img = img2scene(path)
  # random crop
  img = randomCrop(img, img_size[0], img_size[1])
  #orig = img
  img = resize(img, img_size)
  # downsample to 16x16 res (aka blur the image)
  croppedImg = downsample(img, shrink_size, img_size)
  # edge detect
  #croppedImg = edgedetect(croppedImg, orig)
  croppedImg = edgedetect(croppedImg, img)
  # normalize
  img = img2tensor(img)
  croppedImg = img2tensor(croppedImg)
  return img, croppedImg

def save(img, fname="foo.jpg"):
  cv2.imwrite(fname, img)

from glob import glob

def paths(l):
  r = []
  for x in l:
    for path in glob(x):
      r.append(path)
  return r

def main():
  imgs = paths([
    "dataset/*.jpg",
    ])

  imgSize = [1, 1]
  imgSize = [4, 3]
  imgSize = [1280, 720]
  targetSize = [24, 24]

  w, h = imgSize[0], imgSize[1]
  a = w / h
  targetSize[0] = int(a*targetSize[0])

  imgSize = tuple(imgSize)
  targetSize = tuple(targetSize)
  img, cropped = loadRandomImage(imgs, img_size=imgSize, shrink_size=targetSize)
  save(scene2img(cropped), "cropped.jpg")
  save(scene2img(img), "orig.jpg")

if __name__ == '__main__':
  main()
