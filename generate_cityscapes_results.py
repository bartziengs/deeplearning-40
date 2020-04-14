import os
import glob
from PIL import Image

from options.base_options import BaseOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

help_msg = """
The [gtFine_trainvaltest] and [leftImg8bit_trainvaltest] should be present.
gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory. 
leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory. 
The processed images will be placed at --output_dir.

Example usage:

python generate_cityscapes_results.py --gitFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir ./results/cityscapes_predictions/ --dataroot . --direction BtoA --model pix2pix --name cityscapes
"""


def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((256, 256))


# Test options class partly copied from the one test.py uses
class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--gtFine_dir', type=str, required=True,
                            help='Path to the Cityscapes gtFine directory.')
        parser.add_argument('--leftImg8bit_dir', type=str, required=True,
                            help='Path to the Cityscapes leftImg8bit_trainvaltest directory.')
        parser.add_argument('--output_dir', type=str, required=True,
                            default='./results/cityscapes_predictions',
                            help='Directory the output images will be written to.')

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


def main():
    # Set and parse options
    opt = TestOptions().parse()
    opt.name = 'cityscapes'
    opt.model = 'pix2pix'
    opt.direction = 'BtoA'
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # Dataset is the difficult point, as it expects {test, train} A and B folders
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # Create output directories for images.
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.output_dir + 'frankfurt', exist_ok=True)
    os.makedirs(opt.output_dir + 'lindau', exist_ok=True)
    os.makedirs(opt.output_dir + 'munster', exist_ok=True)
    print("Directory structure prepared at %s" % opt.output_dir)

    # create a web page to display results
    web_dir = os.path.join(opt.output_dir, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: researchers use batchnorm and dropout in the original pix2pix.
    # You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()

    # Find all images
    photo_expr = os.path.join(opt.leftImg8bit_dir, opt.phase) + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    print("%d images that match [%s]" % (len(photo_paths), photo_expr))

    # Loop through all photo paths
    for i, photo_path in enumerate(photo_paths):
        photo = load_resized_img(photo_path)

        savepath = os.path.join(opt.output_dir, "%d.png" % i)
        if i % (len(photo_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(photo_paths), savepath))
        pass

    # Normal test code from test.py
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML


if __name__ == '__main__':
    main()
