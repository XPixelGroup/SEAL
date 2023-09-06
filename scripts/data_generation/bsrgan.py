import numpy as np
import cv2
import torch
from os import path as osp
# import bsre.models.utils_image as util
from utils_image import * 

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth


def add_bsrgan(image, degradation_para):

    deg_para = degradation_para.split('|')
    out = degradation_bsrgan(image, deg_para) * 255

    return out


def degradation_bsrgan(img, degradation_para):
    """
    This is the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    isp_model: camera ISP model

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    isp_model = None
    shuffle_order = degradation_para[0]

    dp_pre = degradation_para[1].split('_')

    dp0 = degradation_para[2].split('_')

    dp1 = degradation_para[3].split('_')

    dp2 = degradation_para[4].split('_')

    dp3 = degradation_para[5].split('_')

    dp4 = degradation_para[6].split('_')

    dp5 = degradation_para[7].split('_')

    dp7 = degradation_para[8].split('_')


    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25

    sf = 4

    rnum1 = float(dp_pre[2])
    rnum2 = float(dp_pre[4])

    if sf == 4 and rnum1 < scale2_prob:  # downsample1
        if rnum2 < 0.5:
            img = cv2.resize(
                img, (int(1 / 2 * img.shape[1]), int(1 / 2 * img.shape[0])), interpolation=int(dp_pre[6]))
        else:
            # img = util.imresize_np(img, 1 / 2, True)
            img = imresize_np(img, 1 / 2, True)

        img = np.clip(img, 0.0, 1.0)
        sf = 2

    shuffle_order = shuffle_order.split('_')[1].split('[')[1].split(']')[0].split(', ')

    for i in shuffle_order:

        i = int(i)
        if i == 0:
            img = add_blur(img, dp0, sf=sf)

        elif i == 1:
            img = add_blur(img, dp1, sf=sf)

        elif i == 2:
            a, b = img.shape[1], img.shape[0]
            rnum = float(dp2[2])
            blur_sigma = float(dp2[8])
            sf1 = float(dp2[4])
            # downsample2
            if rnum < 0.75:
                # sf1 = random.uniform(1, 2 * sf)
                interptype = int(dp2[6])
                img = cv2.resize(
                    img, (int(1 / sf1 * img.shape[1]), int(1 / sf1 * img.shape[0])), interpolation=interptype)
            else:

                k = fspecial('gaussian', 25, blur_sigma)
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()  # blur with shifted kernel
                img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]  # nearest downsampling
            img = np.clip(img, 0.0, 1.0)

        elif i == 3:
            # downsample3
            interptype = int(dp3[2])

            img = cv2.resize(img, (int(1 / sf * a), int(1 / sf * b)), interpolation=interptype)

            img = np.clip(img, 0.0, 1.0)


        elif i == 4:
            # add Gaussian noise
            img = add_Gaussian_noise(img, dp4, noise_level1=2, noise_level2=25)

        elif i == 5:
            # add JPEG noise
            rnum = float(dp5[2])

            if rnum < jpeg_prob:
                img = add_JPEG_noise(img, float(dp5[4]))

        elif i == 6:
            # add processed camera sensor noise
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # add final JPEG compression noise
    img = add_JPEG_noise(img, int(dp7[2]))

    return img


def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    '''
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
    k = k.repeat(1, c, 1, 1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n * c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x


def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5 * (scale_factor - 1)  # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)


"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    '''
    # x = util.imresize_np(x, scale=1 / sf)
    x = imresize_np(x, scale=1 / sf)

    return x


def srmd_degradation(x, k, sf=3):
    ''' blur + bicubic downsampling

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):
    ''' bicubic downsampling + blur

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    '''
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, dp, sf=4):
    # wd2 = 4.0 + sf
    # wd = 2.0 + 0.2 * sf

    rnum = float(dp[2])
    ksize = int(dp[4])
    theta = float(dp[6])
    # print(dp)
    if rnum < 0.5:
        l1 = float(dp[8])
        l2 = float(dp[10])
        k = anisotropic_Gaussian(ksize, theta, l1=l1, l2=l2)
    else:
        l3 = float(dp[12])
        k = fspecial('gaussian', ksize, l3)
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0
    intep_type = random.choice([1, 2, 3])
    img = cv2.resize(img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])), interpolation=intep_type)
    img = np.clip(img, 0.0, 1.0)


    return img


def add_Gaussian_noise(img, dp, noise_level1=2, noise_level2=25):
    noise_level = float(dp[4])
    rnum = float(dp[2])

    D = np.diag(np.random.rand(3))
    U = np.random.rand(3, 3)
    count = 6
    for i in range(3):
        for j in range(3):
            D[i][j] = float(dp[count])
            count = count + 1
    count = 16
    for i in range(3):
        for j in range(3):
            U[i][j] = float(dp[count])
            count = count + 1

    if rnum > 0.6:  # add color Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:  # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:  # add  noise
        L = noise_level2 / 255.
        # D = np.diag(np.random.rand(3))
        U = orth(U)
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0, 0, 0], np.abs(L**2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return img


def add_speckle_noise(img, dp, noise_level1=2, noise_level2=25):
    # noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    # rnum = random.random()
    # D = np.diag(np.random.rand(3))
    # U = np.random.rand(3, 3)
    noise_level = float(dp[4])
    rnum = float(dp[2])
    D = np.diag(np.random.rand(3))
    U = np.random.rand(3, 3)
    count = 6
    for i in range(3):
        for j in range(3):
            D[i][j] = float(dp[count])
            count = count + 1
    count = 16
    for i in range(3):
        for j in range(3):
            U[i][j] = float(dp[count])
            count = count + 1
    if rnum > 0.6:
        img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        # D = np.diag(np.random.rand(3))
        U = orth(U)
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L**2 * conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2 * random.random() + 2.0)  # [2, 4]
    rnum = random.random()

    if rnum < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    # deg_para = f'rnum_{rnum}_vals_{vals}'

    return img


def add_JPEG_noise(img, dp):

    quality_factor = int(dp)
    # print(quality_factor)
    # img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)

    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    # img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)

    return img
