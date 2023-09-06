import os
from os import path as osp
import utils_image as util
from util import *

import cv2
import os
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir
from bsrgan import add_bsrgan
from realesrgan import add_realesrgan

def main():

    opt = {}
    opt['n_thread'] = 20

    # set the path of test set
    opt['input_folder'] = 'datasets/Set14-GTmod12'
    save_folder = 'Set14_LQ_SE'

    deg_path = 'scripts/data_generation/degradation_center_100.txt' 
    deg_para = []

    with open(deg_path, 'r') as fp:
        for line in fp:
            x1, deg_para = line.strip().split('/')
            deg_para.append(deg_para)

    for i in range(100):
        if 100 > i > -1:
            opt['deg_para'] = deg_para[i]
            opt['save_folder'] = 'datasets/' + f'{save_folder}/deg_{i}'
            add_degradation(opt)
#------------------------------------------------------------------------------------------

def add_degradation(opt):

    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')

    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

def worker(path, opt):

    img_name, extension = osp.splitext(osp.basename(path))
    LQ_path = opt['save_folder']
    img = util.read_img(path)
    degradation_type = opt['deg_para'].split('_')[0]

    if degradation_type == 'shuffleorder':

        out = add_bsrgan(img, opt['deg_para'])

    else:
        out = add_realesrgan(path, opt['deg_para'])
    
    
    cv2.imwrite(LQ_path + f'/{img_name}.png', out)
    process_info = f'Processing {img_name} ...'
    return process_info



if __name__ == '__main__':
    main()
