# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from monodepth2.networks import ResnetEncoder, DepthDecoder

from monodepth2.layers import disp_to_depth # Not being used atm

from monodepth2.utils import download_model_if_doesnt_exist
import cv2

def load_mondepthWeight(model_name='mono+stereo_1024x320'):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)

    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("-> Loading pretrained decoder")
    depth_decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, (feed_width, feed_height), device


def get_depthDispar(input_image, model=None, modelConfig=None, device='cpu'):

    if model is None:
        raise Exception('Model is empty')
    if input_image is None:
        raise Exception('Input to MonoDepth is empty')

    input_image = pil.fromarray(input_image).convert('RGB')
    encoder, depth_decoder = model

    with torch.no_grad():
        original_width, original_height = input_image.size
        input_image = input_image.resize(modelConfig, pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width),
                                                       mode="bilinear",
                                                       align_corners=False)

        # Unload from GPU to CPU and put it back in numpy
        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
        scaled_disp = scaled_disp.cpu().numpy()

        disp_resized = disp_resized.cpu().numpy()
        disp_resized = disp_resized.squeeze()

        # Covert to colorMap
        vmax = np.percentile(disp_resized, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='inferno')
        colormapped_im = (mapper.to_rgba(disp_resized)[:, :, :3] * 255).astype(np.uint8)

        im = pil.fromarray(colormapped_im)
        #im.save('tokyoS.jpeg')
        #print(disp_resized.shape, scaled_disp.shape, colormapped_im.shape)
        return disp_resized, scaled_disp, colormapped_im, im

def get_depthdispar_simplified(input_image, model=None, modelConfig=None, device='cpu', channel_last=True, isPil=True):

    if model is None:
        raise Exception('Model is empty')
    if input_image is None:
        raise Exception('Input to MonoDepth is empty')

    if not channel_last:
        input_image = np.moveaxis(input_image, 0, 2)
    
    if not isPil:
        input_image = pil.fromarray(input_image).convert('RGB')
    
    encoder, depth_decoder = model

    with torch.no_grad():
        original_width, original_height = input_image.size
        input_image = input_image.resize(modelConfig, pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width),
                                                       mode="bilinear",
                                                       align_corners=False)

        # Unload from GPU to CPU and put it back in numpy
        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
        scaled_disp = scaled_disp.cpu().numpy()

        disp_resized = disp_resized.cpu().numpy()
        disp_resized = disp_resized.squeeze()

        # Covert to colorMap
        vmax = np.percentile(disp_resized, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='inferno')
        colormapped_im = (mapper.to_rgba(disp_resized)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        return disp_resized, scaled_disp, colormapped_im, im

def only_get_disparity(input_image, model=None, modelConfig=None, device='cpu', isPil=True ,channel_last=True):

    if model is None:
        raise Exception('Model is empty')
    if input_image is None:
        raise Exception('Input to MonoDepth is empty')

    if not channel_last:
        input_image = np.moveaxis(input_image, 0, 2)
    
    if not isPil:
        input_image = pil.fromarray(input_image).convert('RGB')
    
    encoder, depth_decoder = model

    with torch.no_grad():
        original_width, original_height = input_image.size
        input_image = input_image.resize(modelConfig, pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                    (original_height, original_width),
                                                    mode="bilinear",
                                                    align_corners=False)

        # Unload from GPU to CPU and put it back in numpy

        disp_resized = disp_resized.cpu().numpy()
        disp_resized = disp_resized.squeeze()

        disp_normalised = (disp_resized - np.min(disp_resized))/np.ptp(disp_resized)
        return disp_normalised


def get_depthdispar_simplified(input_image, model=None, modelConfig=None, device='cpu', channel_last=True, isPil=True):

    if model is None:
        raise Exception('Model is empty')
    if input_image is None:
        raise Exception('Input to MonoDepth is empty')

    if not channel_last:
        input_image = np.moveaxis(input_image, 0, 2)
    
    if not isPil:
        input_image = pil.fromarray(input_image).convert('RGB')
    
    encoder, depth_decoder = model

    with torch.no_grad():
        original_width, original_height = input_image.size
        input_image = input_image.resize(modelConfig, pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                       (original_height, original_width),
                                                       mode="bilinear",
                                                       align_corners=False)

        # Unload from GPU to CPU and put it back in numpy
        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
        scaled_disp = scaled_disp.cpu().numpy()

        disp_resized = disp_resized.cpu().numpy()
        disp_resized = disp_resized.squeeze()

        # Covert to colorMap
        vmax = np.percentile(disp_resized, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='inferno')
        colormapped_im = (mapper.to_rgba(disp_resized)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        return disp_resized, scaled_disp, colormapped_im, im

def get_disparity_Torch_Tensor(input_image, model=None, modelConfig=None, device='cpu', isPil=True ,channel_last=True):

    if model is None:
        raise Exception('Model is empty')
    if input_image is None:
        raise Exception('Input to MonoDepth is empty')

    if not channel_last:
        input_image = np.moveaxis(input_image, 0, 2)
    
    if not isPil:
        input_image = pil.fromarray(input_image).convert('RGB')
    
    encoder, depth_decoder = model

    with torch.no_grad():
        original_width, original_height = input_image.size
        input_image = input_image.resize(modelConfig, pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp,
                                                    (original_height, original_width),
                                                    mode="bilinear",
                                                    align_corners=False)

        # Unload from GPU to CPU and put it back in numpy

        disp_resized = disp_resized.squeeze()
        disp_normalised = (disp_resized - torch.min(disp_resized))/(torch.max(disp_resized)-torch.min(disp_resized)) 
        return disp_normalised
    
if __name__ == '__main__':
    pass
