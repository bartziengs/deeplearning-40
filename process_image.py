import os
import glob
from PIL import Image
import argparse

from options.base_options import BaseOptions
from models import create_model
from data.base_dataset import get_params, get_transform
# import torchvision.transforms as transforms
# import torch

help_msg = """
Under construction

Example usage:

python process_image.py --segmap_path ./gtFine/... --photo_path ./leftImg8bit/... --output_dir ./results/cityscapes_predictions/ --dataroot . --direction BtoA --model pix2pix --name cityscapes
"""


def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))


def process_image(model, segmap_path, photo_path):
    # photo = load_resized_img(photo_path)
    # segmap = load_resized_img(segmap_path)

    # Load images (A = photo, B = segmap)
    A = Image.open(photo_path).convert('RGB')
    B = Image.open(segmap_path).convert('RGB')

    # Transform code stolen from aligned_dataset.py
    # This also includes the resize
    transform_params = get_params(opt, A.size)
    A_transform = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1))
    B_transform = get_transform(opt, transform_params, grayscale=(opt.output_nc == 1))

    A = A_transform(A)
    B = B_transform(B)

    # Unsqueeze is used to wrap the image tensor, as somehow happens in
    # _dataset_fetcher.fetch(index) in torch > utils > data > dataloader.py:385
    data = {'A': A.unsqueeze(0), 'A_paths': [''], 'B': B.unsqueeze(0), 'B_paths': ['']}
    model.set_input(data)
    # Run inference
    model.test()
    # Get image results
    visuals = model.get_current_visuals()

    return visuals


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--segmap_path', required=True, help='path to the segmap label image')
        parser.add_argument('--photo_path', required=True, help='path to the photo image')
        parser.add_argument('--output_dir', type=str, required=True,
                            default='./results/cityscapes_predictions',
                            help='Directory the output image will be written to.')

        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        # Dropout and Batchnorm has different behaviour during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


# Not finished yet
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Set and parse options
    opt = TestOptions().parse()
    opt.name = 'cityscapes'
    opt.model = 'pix2pix'
    opt.direction = 'BtoA'
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    # Dataset is the difficult point, as it expects {test, train} A and B folders
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    process_image(model, opt.segmap_path, opt.photo_path)
