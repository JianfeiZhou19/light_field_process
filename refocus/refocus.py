import os
import sys
import numpy as np

sys.path.append('..')
from utils import load_mat

class Refocus(object):
    def __init__(self, scene, scope, ) -> None:
        super().__init__()
        self.scene = scene
        self.scope = scope

    def _read_data(self):
        raw_data = load_mat(self.scene)
        u, v, h, w, _ = raw_data.shape
        image_ = np.zeros((u*v, h, w, 3))
        for i in range(u):
            for j in range(v):
                image_[i*u+v] = raw_data[i, j]
        return {'image':image_,
                'shape':(u, v, h, w)}
    
    def refocue(self):
        data_dict = self._read_data()
        u, v, h, w = data_dict['shape']
        LF_Image = data_dict['image']
        del data_dict
        shifted_Imr = np.zeros(u*v, h, w)
        shifted_Img = np.zeros(u*v, h, w)
        shifted_Imb = np.zeros(u*v, h, w)
        VVec = np.linspace(-0.5,0.5,u)*(u-1)*self.scope
        UVec =  np.linspace(-0.5,0.5,v)*(v-1)*self.scope
        UMat = np.repeat(UVec, u, 1)
        VMat = np.repeat(VVec, 1, v)
        D = np.zeros(u*v, 2)
        for i in range(u*v):
            D[i, :] = [VMat[i], UMat[i]]
        for i in range(u*v):
            shifted_Imr[i ,:, :] = self.ImWarp(np.squeeze(LF_Image[i,:,:,1]), -D(i,1), -D(i,2))
            shifted_Img[i ,:, :] = self.ImWarp(np.squeeze(LF_Image[i,:,:,2]), -D(i,1), -D(i,2))
            shifted_Imb[i ,:, :] = self.ImWarp(np.squeeze(LF_Image[i,:,:,3]), -D(i,1), -D(i,2))
        refocused_Im = np.zeros(h, w, 3)
        refocused_Im[:, :, 0] = np.sum(shifted_Imr, 1)/u*v
        refocused_Im[:, :, 1] = np.sum(shifted_Img, 1)/u*v
        refocused_Im[:, :, 2] = np.sum(shifted_Imb, 1)/u*v
        return refocused_Im

    def ImWarp(self, img_c, x, y):
        sm = max(abs(int(x))+1, abs(int(y))+1)
        Mask = np.zeros(2*sm+1, 2*sm+1)
        X = int(x) 
        fx = x-X
        Y = int(y) 
        fy = y-Y
        Center = sm + 1
        Mask[Center+Y, Center+X] = (1-fx)*(1-fy)
        Mask[Center+Y, Center+X+1] = fx*(1-fy)
        Mask[Center+Y+1, Center+X] = fy*(1-fx)
        Mask[Center+Y+1, Center+X+1] = fx*fy
        OpIm = img_c*Mask
        return OpIm