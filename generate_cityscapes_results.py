import os
import glob
from PIL import Image

from options.base_options import BaseOptions
# from data import create_dataset
from models import create_model
# from util.visualizer import save_images
from util import html
from data.base_dataset import get_params, get_transform
# import torchvision.transforms as transforms
# import torch
from process_image import process_image

help_msg = """
The [gtFine_trainvaltest] and [leftImg8bit_trainvaltest] should be present.
gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory. 
leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory. 
The processed images will be placed at --output_dir.

Example usage:

python generate_cityscapes_results.py --gitFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir ./results/cityscapes_predictions/ --dataroot . --direction BtoA --model pix2pix --name cityscapes
"""

segmap_postfix = "_gtFine_color.png"
photo_postfix = "_leftImg8bit.png"


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


class MyHTML(html.HTML):
    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        from dominate.tags import meta, h3, table, tr, td, p, a, img, br

        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)


def save_images(webpage, visuals, image_number, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    import util.util

    image_dir = webpage.get_image_dir()
    name = image_number

    webpage.add_header(name)
    ims, txts, links = [], [], []

    postfix = {'real_A': segmap_postfix, 'fake_B': photo_postfix, 'real_B': photo_postfix[:-4] + '_real.png'}

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = name + postfix[label]
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(postfix[label])
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


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
    web_dir = opt.output_dir  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = MyHTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    webpage.img_dir = web_dir

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: researchers use batchnorm and dropout in the original pix2pix.
    # You can experiment it with and without eval() mode.
    if opt.eval:
        model.eval()

    # Find all images
    segmap_base_dir = os.path.join(opt.gtFine_dir, opt.phase)
    photo_base_dir = os.path.join(opt.leftImg8bit_dir, opt.phase)

    segmap_expr = segmap_base_dir + "/*/*_color.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    photo_expr = photo_base_dir + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    assert len(segmap_paths) == len(photo_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (
        len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

    image_numbers = [photo_path[len(photo_base_dir)+1:-len(photo_postfix)] for i, photo_path in enumerate(photo_paths)]

    print("%d images that match [%s]" % (len(photo_paths), photo_expr))

    # Loop through all photo paths
    for i, (segmap_path, photo_path, image_number) in enumerate(zip(segmap_paths, photo_paths, image_numbers)):
        savepath = os.path.join(opt.output_dir, "%s.png" % image_number)

        # # photo = load_resized_img(photo_path)
        # # segmap = load_resized_img(segmap_path)
        #
        # # Load images (A = photo, B = segmap)
        # A = Image.open(photo_path).convert('RGB')
        # B = Image.open(segmap_path).convert('RGB')
        #
        # # Transform code stolen from aligned_dataset.py
        # # This also includes the resize
        # transform_params = get_params(opt, A.size)
        # A_transform = get_transform(opt, transform_params, grayscale=(opt.input_nc == 1))
        # B_transform = get_transform(opt, transform_params, grayscale=(opt.output_nc == 1))
        #
        # A = A_transform(A)
        # B = B_transform(B)
        #
        #
        # # Unsqueeze is used to wrap the image tensor, as somehow happens in
        # # _dataset_fetcher.fetch(index) in torch > utils > data > dataloader.py:385
        # data = {'A': A.unsqueeze(0), 'A_paths': [savepath], 'B': B.unsqueeze(0), 'B_paths': [savepath]}
        # model.set_input(data)
        # model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # # img_path = model.get_image_paths()     # get image paths

        visuals = process_image(model, segmap_path, photo_path)

        save_images(webpage, visuals, image_number, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        if i % 10 == 0:
            webpage.save()  # save the HTML
            print("%d / %d: last image saved at %s, " % (i, len(photo_paths), savepath))

    # Normal test code from test.py
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()           # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()     # get image paths
    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)


if __name__ == '__main__':
    main()
