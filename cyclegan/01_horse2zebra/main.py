import os
import argparse
import torch

from models import ResNetGenerator
import utils


def args_initialize():
    parser = argparse.ArgumentParser()

    parser.add_argument('imfile', help='target image file path')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)

    args = parser.parse_args()

    return args


def main():

    # Get Arguments
    args = args_initialize()
    args.isTrain = False

    # Define Model
    net_G = ResNetGenerator(
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        ngf=args.ngf,
        n_blocks=9
    )

    # Load Weights
    state_dict = torch.load('./latest_net_G.pth', map_location='cpu')
    net_G.load_state_dict(state_dict)

    # Create Tensor from Image file
    im_file = args.imfile
    tensor_img = utils.create_data(im_file)

    # Predict
    outputs = net_G.forward(tensor_img)

    # Convert Output Tensor to Image file
    im = utils.tensor2im(outputs)
    save_path = os.path.join('./test_main.png')
    utils.save_image(im, save_path)


if __name__ == '__main__':
    main()