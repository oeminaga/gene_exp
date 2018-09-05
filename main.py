import keras
from openslide import OpenSlide
from sklearn.externals import joblib

from model import GeneExpressionLevel
import utils
import tensorflow as tf_X
import keras
import metrics
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import numpy as np
import os
import loss_functions
import keras.backend as K
import analyseimage
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
def GeneratePatchImages(args):
    print("Proc: Running the patch images")
    file_manager = utils.Filedirectoryamagement(mask_directories=None, image_directories=args.source)
    file_manager.generate_patch_images_train_valid_test(patch_per_image=128, image_patch_size=(512,512), directory=args.destination, class_filename="sample_cnv.csv", type_class_col="Cluster")
    print("Done: Patch processing...")

def train(args, model, Load_numpy=False, multi_gpu=False, load_augmented_data=False, load_mask=False, class_mode="categorical"):
    print('-' * 30 + 'Begin: training ' + '-' * 30)
    train_data_datagen = ImageDataGenerator()#horizontal_flip=True, zoom_range=0.2, rotation_range=90., shear_range=0.2)
    valid_data_datagen = ImageDataGenerator()#horizontal_flip=True, zoom_range=0.2, rotation_range=90., shear_range=0.2)
    seed = 1
    # Prepare generators..
    if (load_augmented_data):
        print("loading the previous augmented data...")
        valid_img_dataset = np.load('./test_augmented_data_img.npy')
        valid_mask_dataset = np.load('./test_augmented_data_mask.npy')
        train_img_dataset = np.load('./train_augmented_data_img.npy')
        train_mask_dataset = np.load('./train_augmented_data_mask.npy')
        print("training set: ", train_img_dataset.shape)
        print("validation set: ", valid_img_dataset.shape)
        train_input_generator = train_data_datagen.flow(train_img_dataset, train_mask_dataset, batch_size=args.batch_size)
        valid_input_generator = valid_data_datagen.flow(valid_img_dataset, valid_mask_dataset, batch_size=args.batch_size)
    elif (Load_numpy):
        print("Proc: Loading the image list...")
        train_img_dataset = np.load('./train_img_dataset_32.pkl.npy')
        train_mask_dataset = np.load('./train_mask_dataset_32.pkl.npy')
        train_img_dataset, train_mask_dataset = utils.PrepareData(train_img_dataset, train_mask_dataset, "train",
                                                                     args.input_shape,
                                                                     color_normalization=args.color)
        print("train image number:", train_img_dataset.shape[0])
        valid_img_dataset = np.load('./valid_img_dataset_32.pkl.npy')
        valid_mask_dataset = np.load('./valid_mask_dataset_32.pkl.npy')
        valid_img_dataset, valid_mask_dataset = utils.PrepareData(valid_img_dataset, valid_mask_dataset, "test",
                                                                     args.target_size,
                                                                     color_normalization=args.color)
        print("validation image number:", valid_img_dataset.shape[0])
        train_data_datagen = ImageDataGenerator()
        valid_data_datagen = ImageDataGenerator()
        print(train_img_dataset.shape)
        train_input_generator = train_data_datagen.flow(train_img_dataset, train_mask_dataset,
                                                            batch_size=args.batch_size)
        valid_input_generator = valid_data_datagen.flow(valid_img_dataset, valid_mask_dataset,
                                                            batch_size=args.batch_size)

    elif load_mask:
        print("Proc: Generating the image list...")
        train_input_generator = train_data_datagen.flow_from_directory(
                args.path_train + "/input",
                seed=seed,
                mask_directory=args.path_train + "/mask",
                # color_mode="binary",
                equalize_adaphist=False,
                rescale_intensity=False,
                set_random_clipping=True,
                generate_HE=True,
                generate_LAB=True,
                max_image_number=0,
                target_size=args.input_shape,
                batch_size=args.batch_size,
                class_mode='mask')

        validation_input_generator = valid_data_datagen.flow_from_directory(
                args.path_validation + "/input",
                seed=seed,
                mask_directory=args.path_validation + "/mask",
                max_image_number=0,
                equalize_adaphist=False,
                rescale_intensity=False,
                set_random_clipping=True,
                generate_HE=True,
                generate_LAB=True,
                target_size=args.input_shape,
                batch_size=args.batch_size,
                class_mode='mask')
    else:
        print("Proc: Generating the image list...")
        train_input_generator = train_data_datagen.flow_from_directory(
            args.path_train,
            seed=seed,
            mask_directory=None,
            equalize_adaphist=False,
            rescale_intensity=False,
            set_random_clipping=True,
            generate_HE=True,
            generate_LAB=True,
            max_image_number=0,
            target_size=args.input_shape,
            batch_size=args.batch_size,
            class_mode=class_mode)

        validation_input_generator = valid_data_datagen.flow_from_directory(
            args.path_validation,
            seed=seed,
            mask_directory=None,
            max_image_number=0,
            equalize_adaphist=False,
            rescale_intensity=False,
            set_random_clipping=True,
            generate_HE=True,
            generate_LAB=True,
            target_size=args.input_shape,
            batch_size=args.batch_size,
            class_mode=class_mode)

    print("Done: Image lists are created...")
    # callbacks
    print("Proc: Preprare the callbacks...")
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                                   batch_size=args.batch_size, histogram_freq=args.debug)
    #lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_factor,
                                                      patience=args.change_lr_threshold, min_lr=args.min_lr, verbose=1)
    history_register = keras.callbacks.History()

    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_loss',
                                               save_best_only=True, save_weights_only=True, verbose=1,
                                               multi_gpu_mode=multi_gpu, name_of_model="model_1")
    print("Done: callbacks are created...")
    # compile the modelg
    # , self.precision, self.recall, "acc",
    print("Proc: Compile the model...")
    model.compile(optimizer=optimizers.Adam(lr=args.args.lr),
                  loss=[loss_functions.margin_loss, "mse"],
                  metrics={'ucnet': ['acc', metrics.precision, metrics.recall, metrics.dice_coef]},
                  loss_weights=[1., args.lam_recon])

    print("Done: the model was complied...")
    print("Proc: Training the model...")
    # Training with data augmentation
    if Load_numpy:
        train_steps_per_epoch = np.math.ceil(train_img_dataset.shape[0] / args.batch_size)
        valid_steps_per_epoch = np.math.ceil(valid_img_dataset.shape[0] / args.batch_size)

        model.fit_generator(
            generator=utils.image_generator_flow(train_input_generator, reconstruction=False, reshape=False,
                                                    generate_weight=False, run_one_vs_all_mode=False),
            steps_per_epoch=train_steps_per_epoch,
            epochs=args.epochs,
            use_multiprocessing=True,
            validation_steps=valid_steps_per_epoch,
            validation_data=utils.image_generator_flow(valid_input_generator, reconstruction=False, reshape=False,
                                                          generate_weight=False, run_one_vs_all_mode=False),
            callbacks=[log, tb, checkpoint, reduce_lr])  # lr_decay

    else:

        train_steps_per_epoch = np.math.ceil((train_input_generator.samples) / args.batch_size)
        valid_steps_per_epoch = np.math.ceil((validation_input_generator.samples) / args.batch_size)

        model.fit_generator(
            generator=utils.image_generator(train_input_generator, bool(args.use_cropping), args.input_shape,
                                           cropping_size=args.cropping_size),
            steps_per_epoch=train_steps_per_epoch,
            epochs=args.epochs,
            use_multiprocessing=True,
            validation_steps=valid_steps_per_epoch,
            validation_data=utils.image_generator(validation_input_generator, bool(args.use_cropping),
                                                 args.input_shape, cropping_size=args.cropping_size),
            callbacks=[log, tb, checkpoint, reduce_lr]) #lr_decay

    # serialize weights to HDF5
    model.save(args.save_dir + '/trained_model.h5')
    # model.evaluate_generator
    # from utils import plot_log
    # plot_log(args.save_dir + '/log.csv', show=True)
    print('-' * 30 + 'End: training ' + '-' * 30)

    return model

def Run(args, parallel=True):
    GE = GeneExpressionLevel(args)
    if (parallel):
        with tf_X.device('/cpu:0'):
            # config = tf_X.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = 1
            # config.gpu_options.allow_growth = True
            # set_session(tf_X.Session(config=config))
            model, eval_model = GE.CapsuleNetClippedModel(n_class=args.nb_class)
            # model = eval_model
            model.summary()
    else:

        model, eval_model = GE.CapsuleNetClippedModel(n_class=args.nb_class)

    from keras.models import model_from_json, load_model
    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    if args.load_model:
        print(model.summary())
        print("Proc: Loading the previous model...")
        # load json and create model
        loaded_model_json = None
        with open(args.save_dir + '/trained_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        # loaded_model_json = None
        # with open(self.args.save_dir+'/eval_model.json', 'r') as json_file:
        #    loaded_model_json = json_file.read()
        # eval_model = model_from_json(loaded_model_json)
    # Load the weight is available...
    if args.weights is None and args.load_previous_weight:
        file_name_weight = utils.GetLastWeight(args.save_dir)
    else:
        file_name_weight = args.weights

    if file_name_weight is not None:
        model.load_weights(file_name_weight)
    else:
        print('No weights are provided. Will test using random initialized weights.')

    # Show the model
    # Testing..
    if not args.testing:
        plot_model(model, to_file= args.save_dir + '/model.png', show_shapes=True)

        if (parallel):
            from keras.utils import multi_gpu_model
            print("Parallel:")
            multi_model = multi_gpu_model(model, gpus=args.gpu)
            plot_model(multi_model, to_file=args.save_dir + '/model_parallel.png', show_shapes=True)

            multi_model.summary()
            model = train(args=args, model=multi_model, multi_gpu=parallel, load_augmented_data=args.load_augmented_data)
        else:
            model = train(args=args, model=model, multi_gpu=parallel, load_augmented_data=args.load_augmented_data)

        file_name_weight = utils.GetLastWeight(args.save_dir)
        # eval_model = model
        if file_name_weight is not None:
            print("Loading the weight...")
            eval_model.load_weights(file_name_weight, by_name=True)
        test(model=eval_model, Onlive=True,
                  image_source="/home/eminaga/EncryptedData/Challenge/MoNuSeg Training Data/Tissue images/")
    else:
        if file_name_weight is not None:
            print("Loading the weight...")
            eval_model.load_weights(file_name_weight, by_name=True)
        if os.path.exists('./scaler.pkl'):
            scaler = joblib.load('./scaler.pkl')
            print("Loaded existing scale transformeer...")
        eval_model.summary()
        test(model=eval_model, Onlive=True,
                  image_source=args.filename)

def test(model,Onlive, image_source, args):
    if Onlive:
        from os.path import isfile, join
        from os import listdir
        import analyseimage
        print("Proc: Onlive test....")
        print(image_source)

        # img_files = [f for f in listdir(Flags.source_images) if isfile(join(Flags.source_images, f)) and os.path.splitext(f)[1]==".tif"]
        list_of_image_files = [f for f in listdir(image_source) if
                               isfile(join(image_source, f)) and os.path.splitext(f)[1] == ".tif"]
        # self.GenerateFileListFromDirectory(image_source, supported_image_extension)
        print(list_of_image_files)
        id_ = 0
        for file_img in list_of_image_files:
            filname_img = os.path.join(image_source, file_img)
            id_ += 1
            print(id_)
            filname = os.path.basename(file_img)
            if args.use_cropping:
                target_size = args.cropping_size
            else:
                target_size = args.input_shape

            analyseimage.PredictImage(model, filname_img, id_, "./" + filname + "_heatmap.png", 40,
                              normalization_color = args.color, target_size=target_size, nb_class=args.nb_class, batch_size=args.batch_size )
        return print("Done")

if __name__ == "__main__":
    print("Starting...")
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
    parser.add_argument('--nb_class', '-n', default=3, type=int, help="the number of the classes")
    parser.add_argument('--path_validation', default="/home/eminaga/GenHistomic/valid/",
                        help="images for validation")
    parser.add_argument('--path_train', default="/home/eminaga/GenHistomic/train/",
                        help="images for training")
    parser.add_argument('--path_test', default="/home/eminaga/GenHistomic/test/",
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
    parser.add_argument('--use_cropping', default=1, type=int,
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
                        help="The path of the saved weights. Should be specified when testing")

    parser.add_argument('-gp', '--generate_patch', default=False,
                        help='generate patches')

    parser.add_argument('-s', '--source', default="/home/eminaga/EncryptedData/Diagnostic_image/")
    parser.add_argument('-d', '--destination', default="/home/eminaga/GPU_Server/GenHistomic/")

    parser.add_argument('--input_shape', default=(1024,1024), action='store_true', help="Define the input shape of the image")
    parser.add_argument('--cropping_size', default=(64, 64), action='store_true', help="Define the cropping size")
    parser.add_argument('--filename', default="", help="Define the filename for the test")
    parser.add_argument('--change_lr_threshold', default=5, type=int, help="When to change the learning rate. The epoche number is given.")
    #change_lr_threshold
    args = parser.parse_args()

    print(args)

    if args.generate_patch:
        GeneratePatchImages(args)
    else:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            Run(args, parallel=args.parallel)