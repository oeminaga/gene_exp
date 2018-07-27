import keras
from openslide import OpenSlide
import model
import utils
def GeneratePatchImages(arg):
    print("Proc: Running the patch images")
    file_manager = utils.Filedirectoryamagement(None, image_directories=arg.source)
    file_manager.generate_patch_images_train_valid_test(patch_per_image=100, image_patch_size=(512,512), directory=arg.destination, class_filename="sample_cnv.csv", type_class_col="CNV_Status")
    print("Done: Patch processing...")

def train():
    pass

def test():
    pass

def ___main___():
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    parser = argparse.ArgumentParser(description="Capsule Network on histology images.")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--preload_size', default=1600, type=int)
    parser.add_argument('--lr', default=0.1, type=float,
                        help="Initial learning rate")
    parser.add_argument('--nclass', '-n', default=3, type=int, help="the number of the classes")
    parser.add_argument('--path_validation', default="/home/eminaga/Challenges/nucleus/valid",
                        help="images for validation")
    parser.add_argument('--path_train', default="/home/eminaga/Challenges/nucleus/wtrain",
                        help="images for training")
    parser.add_argument('--path_test', default="/Data/ToStudy_PRAD_20x/test/",
                        help="images for test")
    parser.add_argument('--lr_decay', default=0.999, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,  # 0.392
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--n_channels', default=32, type=int,
                        help="Number of channels.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--gpu', default=3, type=float, help="the number of gpus.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--use_cropping', default=0, type=int,
                        help="Apply the cropping function")
    parser.add_argument('-t', '--testing', default=False, action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-l', '--load_model', default=False, action='store_true',
                        help="Load a previous model.")
    parser.add_argument('-lw', '--load_previous_weight', default=False, action='store_true',
                        help="Load a previous trained weight.")
    parser.add_argument('-p', '--parallel', default=False, action='store_true',
                        help="Set parallel computation.")
    parser.add_argument('-c', '--color', default="HE",
                        help="define the color channel")
    # load_augmented_data
    parser.add_argument('-la', '--load_augmented_data', default=False, action='store_true',
                        help="Load augmented data")
    parser.add_argument('--dim_capsule', default=16, type=int,
                        help="dim of capsule")
    parser.add_argument('-w', '--weights', default=None,
                        help="argparseThe path of the saved weights. Should be specified when testing")

    parser.add_argument('-gp', '--generate_patch', default=False, action='store_true',
                        help='generate patches')

    parser.add_argument('s', '-source', default='')
    parser.add_argument('d', '-destination', default='')

    args = parser.parse_args()

    print(args)

    if args.generate_patch:
        GeneratePatchImages(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

        model_core = model.GeneExpressionLevel(directory_image=,
                                  directory_mask=,
                                  input_shape=,
                                  result_directory=,
                                  batch_size=,
                                  epoch=,
                                  presizeloading=)
        model_core.Run()





