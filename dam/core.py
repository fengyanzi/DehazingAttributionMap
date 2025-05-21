import torchvision
import torch
from scipy import stats
import numpy as np
import torch
import cv2
from dam.utils import _add_batch_one, _remove_batch
from dam.utils import grad_norm, IG_baseline, interpolation, isotropic_gaussian_kernel
from tqdm import tqdm


def attr_grad(tensor, h, w, window=8, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2)
    w_grad = torch.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = torch.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    crop = grad[:, :, h: h + window, w: w + window]
    return torch.sum(crop)

def attr_grad_test(tensor, h, w, window=8, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x, :], 2)
    w_grad = torch.pow(tensor[:, :, :, :w_x], 2)
    grad = torch.pow(h_grad[:, :, :, :] + w_grad[:, :, :, :], 1 / 2)
    crop = grad[:, :, h: h + window, w: w + window]
    crop = tensor[:, :, h: h + window, w: w + window]
    return torch.mean(crop)+torch.var(crop)
def attribution_objective(attr_func, h, w, window=16):
    def calculate_objective(image):
        return attr_func(image, h, w, window=window)

    return calculate_objective


def saliency_map_gradient(numpy_image, model, attr_func):
    img_tensor = torch.from_numpy(numpy_image)
    img_tensor.requires_grad_(True)
    result = model(_add_batch_one(img_tensor))
    target = attr_func(result)
    target.backward()
    return img_tensor.grad.numpy(), result






def Path_gradient(noi_image, cl_image, fold, model, attr_objective, path_interpolation_func, cuda=False):
    """
    :param path_interpolation_func:
        return \lambda(\alpha) and d\lambda(\alpha)/d\alpha, for \alpha\in[0, 1]
        This function return pil_numpy_images
    :return:
    """
    if cuda:
        model = model.cuda()
    noi_numpy_image = np.moveaxis(noi_image, 0, 2)
    cl_numpy_image = np.moveaxis(cl_image, 0, 2)

    image_interpolation, lambda_derivative_interpolation = path_interpolation_func(cl_numpy_image, noi_numpy_image,
                                                                                   fold)  # (cv_numpy_image)
    grad_accumulate_list = np.zeros_like(image_interpolation)
    result_list = []
    for i in tqdm(range(image_interpolation.shape[0])):
        img_tensor = torch.from_numpy(image_interpolation[i])
        img_tensor.requires_grad_(True)
        if cuda:
            result = model(_add_batch_one(img_tensor).cuda())  # / 255.0)
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.cpu().numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0
        else:
            result = model(_add_batch_one(img_tensor).half())
            target = attr_objective(result)
            target.backward()
            grad = img_tensor.grad.numpy()
            if np.any(np.isnan(grad)):
                grad[np.isnan(grad)] = 0.0

        grad_accumulate_list[i] = grad * lambda_derivative_interpolation[i]
        result_list.append(result.detach().cpu().numpy())
    results_numpy = np.asarray(result_list)
    return grad_accumulate_list, results_numpy, image_interpolation


def saliency_map_PG(grad_list, result_list):
    final_grad = grad_list.mean(axis=0)
    return final_grad, result_list[-1]


def cloud_sim_path(clear_np_img, cloud_np_img, fold):
    h, w, c = clear_np_img.shape
    image_interpolation = np.zeros((fold, h, w, c))
    lambda_derivative_interpolation = np.zeros((fold, h, w, c))
    alpha_interpolation = np.linspace(1.0, 0, fold + 1)
    for i in range(fold):
        image_interpolation[i] = mix_img(clear_np_img, cloud_np_img, alpha_interpolation[i + 1])
        # cv2.imshow("mix", cv2.cvtColor(image_interpolation[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)
        lambda_derivative_interpolation[i] = mix_img(clear_np_img, cloud_np_img,
                                                     (alpha_interpolation[i + 1] - alpha_interpolation[i]) * fold)
    return np.moveaxis(image_interpolation, 3, 1).astype(np.float32), \
        np.moveaxis(lambda_derivative_interpolation, 3, 1).astype(np.float32)


def mix_img(cloud, clear, alpha):
    return cv2.addWeighted(cloud, alpha, clear, 1 - alpha, 0)
