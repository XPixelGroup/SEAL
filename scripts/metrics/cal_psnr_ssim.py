import argparse
import cv2
import numpy as np
from os import path as osp
import os
import glob
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import bgr2ycbcr, scandir
import pandas as pd
import warnings
from basicsr.metrics import calculate_niqe

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


def cal_lpips(folder_gt, folder_restored):

    suffix = ''

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))
    # img_list = sorted(scandir(folder_gt, recursive=True, full_path=True))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        # print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)

    lpips_val = (sum(lpips_all) / len(lpips_all)).cpu().detach().numpy()

    return lpips_val



def cal_niqe(args):
    txt_path = args.txt_path
    file_name = open(txt_path, 'w')
    niqe_all = []
    img_list = sorted(scandir(args.input, recursive=True, full_path=True))

    for i, img_path in enumerate(img_list):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
        print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)

    print(args.input)
    return round(sum(niqe_all) / len(niqe_all), 3)
    # print(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')


def main(args):

    psnr_all = []
    ssim_all = []
    # niqe_all = []
    # lpips_all = []

    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    # loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    txt_path = args.txt_path
    file_name = open(txt_path, 'w')

    if args.test_y_channel:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if args.suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # calculate PSNR and SSIM--------------------------------------------------------------------------------
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        # print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.3f} dB, \tSSIM: {ssim:.3f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)

        # calculate niqe-----------------------------------------------------------------------------------------
        # img = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED)

        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', category=RuntimeWarning)
        #     niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
        # # print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        # niqe_all.append(niqe_score)    

        # calculate lpips-----------------------------------------------------------------------------------------

        # img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(
        #     np.float32) / 255.

        # img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # # norm to [-1, 1]
        # normalize(img_gt, mean, std, inplace=True)
        # normalize(img_restored, mean, std, inplace=True)

        # lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda())

        # # print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        # lpips_val = np.float64(lpips_val.cpu().detach().numpy())
        # lpips_all.append(lpips_val)

        # file_name.write(f'{i+1:3d} {basename} PSNR {psnr:.3f} SSIM {ssim:.3f} NIQE: {niqe_score:.3f} LPIPS: {lpips_val:.3f}\n')

        file_name.write(f'{i+1:3d} {basename} PSNR {psnr:.3f} SSIM {ssim:.3f}\n')

    file_name.close

    psnr =  round(sum(psnr_all) / len(psnr_all), 3)
    ssim =  round(sum(ssim_all) / len(ssim_all), 3)
    # lpips_val = round(sum(lpips_all) / len(lpips_all), 3)
    # niqe = round(sum(niqe_all) / len(niqe_all), 3)

    print(args.gt)
    print(args.restored)
    # print(f'Average: PSNR: {psnr} dB, SSIM: {ssim}, lpips: {lpips_val}, niqe {niqe}')
    print(f'Average: PSNR: {psnr} dB, SSIM: {ssim}')

    # return psnr, ssim, lpips_val, niqe
    return psnr, ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='gt', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results', help='Path to restored images')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()

    args.test_y_channel = True


    #  ------------------------------------------------------------------------------------------
    args.gt = 'datasets/Set14-GTmod12'
    results_save = 'results/Set14_SE' #  
    model_name_list = ['fsrcnn', 'srresnet']
    csv_save_path = osp.join('scripts/metrics/results', 'Set14_SE.csv')
    # -------------------------------------------------------------------------------------------------

    data = {}

    for i in range(len(model_name_list)):
        model_name = model_name_list[i]

        num = []
        n_class = 100
        results_psnr = []
        results_ssim = []
        # results_niqe = []
        # results_lpips = []

        for i in range(n_class):
            args.restored = osp.join(results_save, model_name, f'deg_{i}')
            args.txt_path = osp.join(results_save, model_name, f'deg_{i}.txt')

            # psnr, ssim, lpipsval, niqe = main(args)
            psnr, ssim = main(args)

            results_psnr.append(psnr)
            results_ssim.append(ssim)
            # results_niqe.append(niqe)
            # results_lpips.append(lpipsval)
            num.append(i)

        data['model_num'] = num
        data[model_name + '_psnr'] = results_psnr
        data[model_name + '_ssim'] = results_ssim
        # data[model_name + '_lpips'] = results_lpips
        # data[model_name + '_niqe'] = results_niqe

    savedata = pd.DataFrame(data)                                          
    savedata.to_csv(csv_save_path)  




