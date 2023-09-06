import argparse
from turtle import Turtle
import cv2
import glob
import os
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srresnet_arch import MSRResNet
from seal.archs.fsrcnn_arch import FSRCNN

from utils import RealESRGANer

def main(args):
    

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    print(args.model_name)
    netscale = 4

    if args.model_name in ['RealESRGAN_x4plus', 'RRDBNet']: 
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join('modelzoo/mz_RRDBNet/', f'{args.model_name}_{args.model_num}.pth')

    elif args.model_name in ['srresnet']:
        model = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4)
        model_path = os.path.join('modelzoo/excellenceline/', f'{args.model_name}_{args.model_num}.pth')

    elif args.model_name in ['fsrcnn']:  
        model = FSRCNN(num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4)
        model_path = os.path.join('modelzoo/acceptanceline/', f'{args.model_name}_{args.model_num}.pth')

    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.bic:
                output, _ = upsampler.bicubic(img, outscale=args.outscale)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='srresnet',
        help=('Model names: srresnet | rrdb | fsrcnn'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--bic', type=int, default=0, help='Use bicubic to the input')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()


    # Configurations
    input_list = ['datasets/Set14_LQ_SE']
    results_save_root_list= ['results/Set14_lines_SE']

    n_class = 100
    args.bic = 0    
    for j in range(len(input_list)):
        input_path = input_list[j]
        results_save_root = results_save_root_list[j]
        for i in range(n_class):
            
            args.input = os.path.join(input_path, f'deg_{i}')
            print(args.input)
    # --------------------  model zoo   -------------------------------
            args.model_name = 'srresnet' 
            args.model_num = str(i)
            args.output = os.path.join(results_save_root, args.model_name, f'deg_{i}')
            main(args)
    # --------------------  model zoo   -------------------------------
            args.model_name = 'fsrcnn' 
            args.model_num = str(i)
            args.output = os.path.join(results_save_root, args.model_name, f'deg_{i}')
            main(args)       
    # # # --------------------  realSR model   -------------------------------
    #         args.model_name = 'rrdb' 
    #         args.model_num = str(i)
    #         args.output = os.path.join(results_save_root, args.model_name, f'm{i}')
    #         main(args)    




