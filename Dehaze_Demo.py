import cv2
import numpy as np
import pandas as pd
import torch
from dam.utils import Tensor2PIL, PIL2Tensor, cv2_to_pil, pil_to_cv2, vis_saliency, vis_saliency_kde, grad_abs_norm, prepare_images, make_pil_grid
from dam.core import  attribution_objective, Path_gradient, attr_grad, cloud_sim_path, attr_grad_test
from dam.core import saliency_map_PG as saliency_map

# import model
from model.dehaze_backbones.xt.decoders.decoder_pre import xT as DehazeXL

if __name__ == '__main__':

    modelpath = r'model/weights/weightforDAMtest.pth'
    cloudimgpath = r'data/dehaze/test1.png'
    clearingpath = r'data/dehaze/clear1.png'
    save_path = './results/test.png'
    w = 132  # The x coordinate of your select patch, 125 as an example
    h = 164  # The y coordinate of your select patch, 160 as an example
    window_size = 32  # Define windoes_size of D

    # No recommend to change the following parameters
    fold = 5
    sigma = 1.2
    l = 9
    alpha = 0.8

    # Load your own model
    model = DehazeXL().to("cuda").eval()


    model.load_state_dict(torch.load(modelpath, weights_only=True), strict=False)
    img_noi, img_cl = prepare_images(cloudimgpath,clearingpath, scale=4)
    tensor_noi = PIL2Tensor(img_noi)[:3]
    tensor_hr = PIL2Tensor(img_cl)[:3]

    draw_img = pil_to_cv2(img_noi)
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 2)
    position_pil = cv2_to_pil(draw_img)

    attr_objective = attribution_objective(attr_grad_test, h, w, window=window_size)

    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_noi.numpy(),tensor_hr.numpy(),fold, model, attr_objective, cloud_sim_path,cuda=True) #(np.array(img_cl), np.array(img_noi), fold)
    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)

    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy)
    #saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)

    #blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_noi.resize(img_cl.size)) * alpha)
    #blend_kde_and_input = cv2_to_pil( pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_noi.resize(img_cl.size)) * alpha)

    pil = make_pil_grid(
        [position_pil,
         saliency_image_abs,
    #     blend_abs_and_input,
         Tensor2PIL(torch.clamp(torch.from_numpy(result), min=0., max=1.))]
    )
    pil.save(save_path)

