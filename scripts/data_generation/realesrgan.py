import numpy as np
import torch
import cv2

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG
from util import *
from scipy.stats import multivariate_normal



def add_realesrgan(path, degradation_para):
    img_HR = cv2.imread(path)
    img_HR = img2tensor(img_HR)
    img_HR = torch.unsqueeze(img_HR, 0)


    img_lq = degradation_realesrgan(img_HR, degradation_para)

    image_LR_blur = tensor2img(img_lq[0])

    return image_LR_blur


def degradation_realesrgan(image, degradation_para):
    num_gpu = 0
    scale = 4

    device = torch.device('cuda' if num_gpu != 0 else 'cpu')

    ori_h, ori_w = image.size()[2:4]

    deg_para = degradation_para.split('_')

    #---------------------------- First Order ----------------------------------
    out = one_order(image, deg_para[0:25], device)

    #---------------------------- 4 jpeg parameters ----------------------------
    First_jpeg_flag = int(deg_para[23])
    First_jpeg_para = deg_para[24]
    jpeg_range_select = float(First_jpeg_para) # np.random.random() * 65 + 30
    jpeg_range = [jpeg_range_select, jpeg_range_select]

    #---------------------------- 4 add jpeg ----------------------------------
    # JPEG compression
    jpeger = DiffJPEG(differentiable=False)
    
    if First_jpeg_flag:
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)

    #---------------------------- Second Order ----------------------------------
    out = one_order(out, deg_para[25:], device)

    # ------------------------------------- Final order sinc kernel parameters ------------------------------------- #
    Final_order= deg_para[49]
    Final_order_para = deg_para[49:57]

    # ------------------------------------- Final order sinc kernel parameters ------------------------------------- #

    sinc_kernel_size = 0
    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1

    if int(Final_order_para[1]):
        sinc_kernel_size = int(Final_order_para[2]) #random.choice(kernel_range)
        sinc_omega_c = float(Final_order_para[3]) #np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(sinc_omega_c, sinc_kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)

    else:

        sinc_kernel = pulse_tensor
        sinc_omega_c = 0

    b, c, h, w = image.size()

    sinc_kernel = torch.FloatTensor(sinc_kernel)
    if b >1:
        n, n = sinc_kernel.size()
        sinc_kernel = sinc_kernel.view(1,1,n,n).repeat(b,1,1,1)

    #---------------------------- jpeg parameters ----------------------------------
    jpeg_range_select = float(Final_order_para[7]) #  np.random.random() * 65 + 30
    jpeg_range = [jpeg_range_select, jpeg_range_select]

    jpeger = DiffJPEG(differentiable=False)


    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    #   3. [resize back + sinc filter]
    #   4. resize back
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.

    final_type_range = [0,1,2,3]
    final_type_range_select = int(Final_order_para[0]) #random.choice(final_type_range)

    mode = Final_order_para[5] #random.choice(['area', 'bilinear', 'bicubic'])


    if final_type_range_select == 0 :
        # resize back + the final sinc filter
        out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
        out = torch.clamp(out, 0, 1).to(device)

        out = jpeger(out, quality=jpeg_p)

    elif final_type_range_select == 1 : 
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)

    elif final_type_range_select == 2 :
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)
        out = filter2D(out, sinc_kernel)

    elif final_type_range_select == 3 :
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // scale, ori_w // scale), mode=mode)

    return out


def one_order(img, Degradation_para, device):

    First_blur_flag = int(Degradation_para[2])
    First_blur_para = Degradation_para[3:12]

    First_resize_flag = int(Degradation_para[13])
    First_resize_para = Degradation_para[14:16]

    First_noise_flag = int(Degradation_para[17])
    First_noise_para = Degradation_para[18:22]

    #---------------------------- kernel setting ----------------------------------
    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1

    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob = [0, 0, 0, 0, 0, 0]

    
    #-----------------------  blur parameters   --------------------#
    blur_sinc_flag = int(First_blur_para[0])
    kernel_size = int(First_blur_para[1])
    # kernel_type_range = [0,1,2,3,4,5]
    kernel_type_range_select = int(First_blur_para[2])
    kernel_prob[kernel_type_range_select] = 1

    base_sig = 3.8 # 2.8
    bias_sig = 0.2 #0.2 0.8 1.4 2.0 2.6

    blur_sigma_select = float(First_blur_para[3]) # np.random.random() * base_sig + bias_sig   
    blur_sigma = [blur_sigma_select, blur_sigma_select]
    blur_sigma_select2 = float(First_blur_para[4]) #np.random.random() * base_sig + bias_sig
    blur_sigma2 = [blur_sigma_select2, blur_sigma_select2]

    betag_select = float(First_blur_para[5]) # np.random.random() * 3.5 + 0.5
    betag_range = [betag_select, betag_select]

    betap_select = float(First_blur_para[6]) #np.random.random() * 1 + 1
    betap_range = [betap_select, betap_select]

    rotate_angle = float(First_blur_para[7]) #np.random.uniform(-math.pi, math.pi)

    omega_c = 0

    #----------------------- 1 blur kernel generation   --------------------#

    if blur_sinc_flag:  #0.5
        
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = float(First_blur_para[8]) # np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = float(First_blur_para[8]) # np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma2, [rotate_angle, rotate_angle],
            betag_range,
            betap_range,
            noise_range=None)
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    b, c, h, w = img.size()


    kernel = torch.FloatTensor(kernel)

    if b >1:
        n, n = kernel.size()
        kernel = kernel.view(1,1,n,n).repeat(b,1,1,1)


    #----------------------- 2 resize parameters   --------------------#


    #----------------------- 3 Noise parameters   --------------------#

    noise_range_select = float(First_noise_para[2]) #np.random.random() * base_sig + bias_sig
    noise_range = [noise_range_select, noise_range_select]

    poisson_scale_range_select = float(First_noise_para[3]) #np.random.random() * 3.95 + 0.05
    poisson_scale_range = [poisson_scale_range_select, poisson_scale_range_select]

    #-----------------------  1 add blur   --------------------#

    if First_blur_flag:
        out = filter2D(img, kernel)
    else:
        out = img

    #-----------------------  2 add resize   --------------------#
    if First_resize_flag:

        scale = float(First_resize_para[0])
        mode =  First_resize_para[1] #random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

    else:
        mode = 'bicubic'
        # resize_flag = 0
        scale = 0.25

        out = F.interpolate(out, scale_factor=0.25, mode='bicubic')

    #---------------------------- 3 add noise ----------------------------------

    if First_noise_flag:

        gray_noise_flag = int(First_noise_para[0])
        gaussian_noise_flag = int(First_noise_para[1])

        if gray_noise_flag: #0.5
            gray_noise_prob = 1
        else:
            gray_noise_prob = 0
        
        if gaussian_noise_flag:  #0.5

            out = random_add_gaussian_noise_pt(
                out, sigma_range=noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:

            out = random_add_poisson_noise_pt(
                out,
                scale_range=poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

    return out



# -------------------------------------------------------------------- #
# --------------------------- blur kernels --------------------------- #
# -------------------------------------------------------------------- #


# --------------------------- util functions --------------------------- #
def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (two dimensional matrix).

    Args:
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.

    Returns:
        ndarray: Rotated sigma matrix.
    """
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size):
    """Generate the mesh grid, centering at zero.

    Args:
        kernel_size (int):

    Returns:
        xy (ndarray): with the shape (kernel_size, kernel_size, 2)
        xx (ndarray): with the shape (kernel_size, kernel_size)
        yy (ndarray): with the shape (kernel_size, kernel_size)
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy


def pdf2(sigma_matrix, grid):
    """Calculate PDF of the bivariate Gaussian distribution.

    Args:
        sigma_matrix (ndarray): with the shape (2, 2)
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.

    Returns:
        kernel (ndarrray): un-normalized kernel.
    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel


def cdf2(d_matrix, grid):
    """Calculate the CDF of the standard bivariate Gaussian distribution.
        Used in skewed Gaussian distribution.

    Args:
        d_matrix (ndarrasy): skew matrix.
        grid (ndarray): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size.

    Returns:
        cdf (ndarray): skewed cdf.
    """
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    grid = np.dot(grid, d_matrix)
    cdf = rv.cdf(grid)
    return cdf


def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel.

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (bool):

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """Generate a bivariate generalized Gaussian kernel.
        Described in `Parameter Estimation For Multivariate Generalized
        Gaussian Distributions`_
        by Pascal et. al (2013).

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.

    .. _Parameter Estimation For Multivariate Generalized Gaussian
    Distributions: https://arxiv.org/abs/1302.6498
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    """Generate a plateau-like anisotropic kernel.
    1 / (1+x^(beta))

    Ref: https://stats.stackexchange.com/questions/203629/is-there-a-plateau-shaped-distribution

    In the isotropic mode, only `sig_x` is used. `sig_y` and `theta` is ignored.

    Args:
        kernel_size (int):
        sig_x (float):
        sig_y (float):
        theta (float): Radian measurement.
        beta (float): shape parameter, beta = 1 is the normal distribution.
        grid (ndarray, optional): generated by :func:`mesh_grid`,
            with the shape (K, K, 2), K is the kernel size. Default: None

    Returns:
        kernel (ndarray): normalized kernel.
    """
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    #assert sigma_x_range[0] < sigma_x_range[1], ' sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        #assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        #assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        #assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_generalized_Gaussian(kernel_size,
                                          sigma_x_range,
                                          sigma_y_range,
                                          rotation_range,
                                          beta_range,
                                          noise_range=None,
                                          isotropic=True):
    """Randomly generate bivariate generalized Gaussian kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    #assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        #assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        #assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # assume beta_range[0] < 1 < beta_range[1]
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_generalized_Gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        #assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_plateau(kernel_size,
                             sigma_x_range,
                             sigma_y_range,
                             rotation_range,
                             beta_range,
                             noise_range=None,
                             isotropic=True):
    """Randomly generate bivariate plateau kernels.

    In the isotropic mode, only `sigma_x_range` is used. `sigma_y_range` and `rotation_range` is ignored.

    Args:
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi/2, math.pi/2]
        beta_range (tuple): [1, 4]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    #assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        #assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        #assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    # TODO: this may be not proper
    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic)
    # add multiplicative noise
    if noise_range is not None:
        #assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)

    return kernel


def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=(0.6, 5),
                         sigma_y_range=(0.6, 5),
                         rotation_range=(-math.pi, math.pi),
                         betag_range=(0.5, 8),
                         betap_range=(0.5, 8),
                         noise_range=None):
    """Randomly generate mixed kernels.

    Args:
        kernel_list (tuple): a list name of kernel types,
            support ['iso', 'aniso', 'skew', 'generalized', 'plateau_iso',
            'plateau_aniso']
        kernel_prob (tuple): corresponding kernel probability for each
            kernel type
        kernel_size (int):
        sigma_x_range (tuple): [0.6, 5]
        sigma_y_range (tuple): [0.6, 5]
        rotation range (tuple): [-math.pi, math.pi]
        beta_range (tuple): [0.5, 8]
        noise_range(tuple, optional): multiplicative kernel noise,
            [0.75, 1.25]. Default: None

    Returns:
        kernel (ndarray):
    """
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=True)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_Gaussian(
            kernel_size,
            sigma_x_range,
            sigma_y_range,
            rotation_range,
            betag_range,
            noise_range=noise_range,
            isotropic=False)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=True)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, betap_range, noise_range=None, isotropic=False)
    return kernel
