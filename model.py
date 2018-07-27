from keras import layers
import keras
import keras_preprocessing
import keras.backend as K
from keras import layers, models, optimizers
from keras.layers.merge import concatenate
from sklearn.externals import joblib
import os
import metrics
import loss_functions
import custom_layers
import numpy as np
import tensorflow as tf_X
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

class GeneExpressionLevel():
    def sampling(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))

        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def __init__(self, directory_image, directory_mask, input_shape, result_directory, batch_size, epoch, cropping_size=None, presizeloading=None):
        self.directory_image = directory_image
        self.directory_mask = directory_mask
        self.input_shape = input_shape
        self.result_directory = result_directory
        self.batch_size = batch_size
        self.epoch = epoch
        self.original_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.patch_size_loading = presizeloading
        self.cropping_size = cropping_size

    def train(self, model):
        print('-' * 30 + 'Begin: training ' + '-' * 30)
        train_data_datagen = ImageDataGenerator()  # rescale=1. / 255)
        valid_data_datagen = ImageDataGenerator()  # rescale=1. / 255)
        seed = 1
        # Prepare generators..
        print("Proc: Generating the image list...")
        train_input_generator = train_data_datagen.flow_from_directory(
            self.path_train + "/input",
            seed=seed,
            mask_directory=self.path_train + "/mask",
            # color_mode="binary",
            equalize_adaphist=False,
            rescale_intensity=False,
            set_random_clipping=True,
            generate_HE=True,
            generate_LAB=True,
            max_image_number=100000,
            target_size=self.target_size,
            batch_size=self.args.preload_size,
            class_mode='mask')

        validation_input_generator = valid_data_datagen.flow_from_directory(
            self.path_validation + "/input",
            seed=seed,
            mask_directory=self.path_validation + "/mask",
            max_image_number=0,
            equalize_adaphist=False,
            rescale_intensity=False,
            set_random_clipping=True,
            generate_HE=True,
            generate_LAB=True,
            target_size=self.target_size,
            batch_size=self.args.preload_size,
            class_mode='mask')
        print("Done: Image lists are created...")

        # callbacks
        print("Proc: Preprare the callbacks...")
        log = callbacks.CSVLogger(self.args.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=self.args.save_dir + '/tensorboard-logs',
                                   batch_size=self.batch_size, histogram_freq=self.args.debug)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: self.args.lr * (0.9 ** epoch))
        checkpoint = callbacks.ModelCheckpoint(self.args.save_dir + '/weights-{epoch:02d}.h5',
                                               monitor='val_capsnet_dice_coef',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        print("Done: callbacks are created...")

        # compile the model
        # , self.precision, self.recall, "acc",
        print("Proc: Compile the model...")
        model.compile(optimizer=optimizers.Adam(lr=self.args.lr),
                      loss=[self.margin_loss, "mse"],
                      metrics={'capsnet': ['acc', self.precision, self.recall, self.dice_coef]},
                      loss_weights=[1., self.args.lam_recon])
        print("Done: the model was complied...")

        print("Proc: Training the model...")
        # Training with data augmentation
        sub_images_number = self.get_number_to_cut(self.cropping_size, self.input_shape)

        train_steps_per_epoch = np.math.ceil((train_input_generator.samples * sub_images_number) / self.args.batch_size)
        valid_steps_per_epoch = np.math.ceil(
            (validation_input_generator.samples * sub_images_number) / self.args.batch_size)
        model.fit_generator(
            generator=self.image_generator(train_input_generator, bool(self.args.use_cropping), self.input_shape,
                                           cropping_size=self.cropping_size),
            steps_per_epoch=train_steps_per_epoch,
            epochs=self.args.epochs,
            validation_steps=valid_steps_per_epoch,
            validation_data=self.image_generator(validation_input_generator, bool(self.args.use_cropping),
                                                 self.input_shape, cropping_size=self.cropping_size),
            callbacks=[log, tb, lr_decay, checkpoint])

        from utils import plot_log
        plot_log(args.save_dir + '/log.csv', show=True)
        print('-' * 30 + 'End: training ' + '-' * 30)

    def test(self, model, data, args):
        test_input_datagen = ImageDataGenerator()  # rescale=1./255)
        test_generator = test_input_datagen.flow_from_directory(
            self.args.path_test + "/input",
            # seed=seed,
            equalize_adaphist=False,
            rescale_intensity=False,
            # set_random_clipping=True,
            generate_HE=True,
            generate_LAB=True,
            mask_directory=self.args.path_test + "/mask",
            max_image_number=1000,
            target_size=self.target_size,
            batch_size=self.args.batch_size,
            class_mode='mask')

        batches = 0
        classes = []
        predicted = []
        class_labels = ["No tumor", "Tumor"]
        counter = 0
        for (x_batch, y_batch, location_batch) in self.image_generator(test_generator, bool(self.args.use_cropping),
                                                                       test_mode=True, self.input_shape,
                                                                       cropping_size=self.cropping_size):
            batch_size = x_batch.shape[0]
            score, x_recon = model.predict(x_batch, batch_size=batch_size, verbose=1)
            predict_construction = np.zeros(
                (self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=float)
            image_construction = np.zeros(
                (self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=float)
            images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], x_batch.shape), dtype=float)

            factor_by = 0

            for itm in location_batch:
                predict_construction[:, itm[0]:itm[1], itm[2]:itm[3],
                :] = score[factor_by * self.batch_size:(factor_by + 1) * self.batch_size, 1]
                factor_by = factor_by + 1

            factor_by = 0
            for itm in location_batch:
                image_construction[:, itm[0]:itm[1], itm[2]:itm[3],
                :] = x_recon[factor_by * self.batch_size:(factor_by + 1) * self.batch_size, 1]
                factor_by = factor_by + 1

            factor_by = 0
            for itm in location_batch:
                image[:, itm[0]:itm[1], itm[2]:itm[3],
                :] = x_recon[factor_by * self.batch_size:(factor_by + 1) * self.batch_size, 1]
                factor_by = factor_by + 1

            predicted_classes = np.argmax(score, axis=1)
            predicted.extend(list(predicted_classes.flatten()))
            classes.extend(list(np.argmax(y_batch_cropped, axis=1).flatten()))
            batches += len(x_batch)
            counter += len(x_batch)
            if batches % 100 == 0:
                print("Images done: %s" % (batches))
            if counter >= 5000:
                print("Between-step performance verification:")
                report = classification_report(classes, predicted, target_names=class_labels)
                img = combine_images(np.concatenate([x_batch[:10], x_recon[:10]]))
                image = img * 255
                Image.fromarray(image.astype(np.uint8)).save(
                    args.save_dir + "/" + str(counter) + "_test_real_and_recon.png")
                # print()
                print('Reconstructed test images are saved to %s/real_and_recon.png' % args.save_dir)
                print(report)
                counter = 0

        print("Final report:")
        report = classification_report(classes, predicted, target_names=class_labels)
        print(report)
        print('-' * 30 + 'End: test' + '-' * 30)

    def CoreModel(self, inputs=None):
        intermediate_dim = 512
        latent_dim = 2

        # VAE model = encoder + decoder
        # build encoder model
        x = layers.Dense(intermediate_dim, activation='relu')(inputs)
        self.z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

        h = layers.Dense(intermediate_dim, activation='relu')(x)
        z_mean = layers.Dense(latent_dim)(h)
        self.z_log_sigma = layers.Dense(latent_dim)(h)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = keras.Model(inputs, [self.z_mean, self.z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
        x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = layers.Dense(self.input_shape, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = keras.Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = keras.Model(inputs, outputs, name='vae_mlp')
        return vae

    def CapsuleNetClippedModel(self, n_class=2, routing=3, dim_capsule=8, n_channels=32):
        shape_default = (self.input_shape[0], self.input_shape[1], 2)

        if (bool(self.args.use_cropping)):
            shape_default = (self.cropping_size[0], self.cropping_size[1], 2)
            x = layers.Input(shape=shape_default)
        else:
            x = layers.Input(shape=shape_default)

        if x is None:
            print("Error: No input was given.")
            return None

        shape_default = (shape_default[0], shape_default[1], 2)
        # Layer 1: Just a batch normal conventional Conv2D layer
        conv1 = layers.Conv2D(
            filters=512,
            kernel_size=5,
            strides=1,
            padding='valid',
            name="conv1",
            activation='selu',
            use_bias=False)(x)

        # self.Conv2DBNSLU(x=x, filters=512, kernel_size=4, strides=1, padding='valid', activation='tanh', name='bn_dbcov1')
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = custom_layers.PrimaryCap(conv1, dim_capsule=int(round(dim_capsule / 2)), n_channels=n_channels, kernel_size=5,
                                 strides=2, padding='valid')
        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = custom_layers.CapsuleLayer(num_capsule=n_class, dim_capsule=dim_capsule, routings=routing, name='digitcaps')(
            primarycaps)
        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = custom_layers.Length(name='capsnet')(digitcaps)

        # Decoder network.
        y = layers.Input(shape=(n_class,))
        masked_by_y = custom_layers.Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = custom_layers.Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(256, activation='selu', input_dim=dim_capsule * n_class))
        decoder.add(layers.Dense(512, activation='selu'))
        decoder.add(layers.Dense(np.prod(shape_default), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=shape_default, name='out_recon'))

        # manipulate model
        noise = layers.Input(shape=(n_class, dim_capsule))

        noised_digitcaps = layers.Add()([digitcaps, noise])
        masked_noised_y = custom_layers.Mask()([noised_digitcaps, y])

        # Models for training and evaluation (prediction)
        train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = models.Model(x, [out_caps, decoder(masked)])

        # manipulate model
        noise = layers.Input(shape=(n_class, dim_capsule))
        noised_digitcaps = layers.Add()([digitcaps, noise])
        masked_noised_y = custom_layers.Mask()([noised_digitcaps, y])
        manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

        return train_model, eval_model, manipulate_model

    def UNetCapsuleNetClippedModel(self, n_class=3, number_of_channel=3):
        shape_default = (self.input_shape[0], self.input_shape[1], number_of_channel)
        x = layers.Input(shape=shape_default)

        # 56x56 Level 3
        n_filter = 512
        conv_level_3 = self.Conv2DBNSLU(x=x, filters=32, kernel_size=1, strides=1,
                                        padding='valid', activation='selu', name='conv_level_3_0')

        print(conv_level_3.shape)
        # 28x28 Level 2
        conv_level_2 = layers.Conv2D(filters=64, kernel_size=2, strides=2,
                                     padding='same', activation='selu', name='conv_level_2')(conv_level_3)
        conv_level_2 = layers.SpatialDropout2D(0.2)(conv_level_2)

        # 14x14
        conv_level_1 = layers.Conv2D(filters=128, kernel_size=2, strides=1, padding='same', activation='selu',
                                     name='conv_level_1')(conv_level_2)
        conv_level_1 = layers.SpatialDropout2D(0.2)(conv_level_1)
        conv_level_1 = layers.MaxPooling2D(pool_size=(2, 2), name='pool_level_1')(conv_level_1)

        # subgroups
        conv_level_1_7 = layers.Conv2D(filters=128, kernel_size=7, strides=1, padding='same', activation='selu',
                                       name='conv_level_1_7')(conv_level_1)
        conv_level_1_7 = layers.SpatialDropout2D(0.2)(conv_level_1_7)

        conv_level_1_5 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='same', activation='selu',
                                       name='conv_level_1_5')(conv_level_1)
        conv_level_1_5 = layers.SpatialDropout2D(0.2)(conv_level_1_5)

        conv_level_1_3 = layers.Conv2D(filters=128, kernel_size=(3, 5), strides=1, padding='same', activation='selu',
                                       name='conv_level_1_3')(conv_level_1)
        conv_level_1_3 = layers.SpatialDropout2D(0.2)(conv_level_1_3)

        # Concatenate
        conv_level_1 = concatenate([conv_level_1,
                                    conv_level_1_5,
                                    conv_level_1_3,
                                    conv_level_1_7])

        # 7x7
        conv_level_0 = layers.Conv2D(filters=256, kernel_size=2, strides=2,
                                     padding='valid', activation='selu', name='conv_level_0')(conv_level_1)

        # Contour detection
        conv_level_0_31 = layers.Conv2D(filters=64, kernel_size=(3, 1), strides=1,
                                        padding='same', activation='selu', name='conv_level_0_13')(conv_level_0)
        conv_level_0_31 = layers.SpatialDropout2D(0.2)(conv_level_0_31)

        conv_level_0_3 = layers.Conv2D(filters=64, kernel_size=3, strides=1,
                                       padding='same', activation='selu', name='conv_level_0_3')(conv_level_0)
        conv_level_0_3 = layers.SpatialDropout2D(0.2)(conv_level_0_3)

        conv_level_0_5 = layers.Conv2D(filters=64, kernel_size=5, strides=1,
                                       padding='same', activation='selu', name='conv_level_0_5')(conv_level_0)
        conv_level_0_5 = layers.SpatialDropout2D(0.2)(conv_level_0_5)

        conv_level_0 = concatenate([conv_level_0, conv_level_0_3, conv_level_0_31, conv_level_0_5])

        # Upsampling from level 0 to level 1; 7 --> 14
        upsample_level_0_to_1 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(conv_level_0)

        conv_level_0_to_1 = concatenate([upsample_level_0_to_1, conv_level_1])

        # Upsampling from level 1 to level 2; 14 --> 28
        upsample_level_1_to_2 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(conv_level_0_to_1)
        conv_level_0_to_2 = concatenate([upsample_level_1_to_2, conv_level_2])
        # Reduce the number of filter by half
        # filter_size = int(conv_level_0_to_2.shape[3])
        # filter_size = int(round(filter_size * 0.8))
        # filter_size = filter_size - int(conv_level_3.shape[3])
        # reduction_level = layers.Conv2D(filter_size, kernel_size=1, strides=1, padding='same', activation='relu', name='reduction_level')(conv_level_0_to_2)

        # Upsampling from level 2 to level 3; 28 --> 56
        upsample_level_2_to_3 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(conv_level_0_to_2)

        conv_level_0_to_2_to_3 = concatenate([upsample_level_2_to_3, conv_level_3])

        filter_layers = layers.Conv2D(filters=n_filter, kernel_size=3, strides=1,
                                      padding='same', activation='selu', name='filter_layer')(conv_level_0_to_2_to_3)

        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1,
                                      padding='same', activation='softmax', name='ucnet')(filter_layers)

        y = layers.Input(shape=(self.input_shape[0], self.input_shape[1], n_class))

        reshaped_filter_layers = layers.Reshape((n_filter, self.input_shape[0] * self.input_shape[1]),
                                                input_shape=(self.input_shape[0], self.input_shape[1], n_filter))(
            filter_layers)

        # train condition
        mask_with_y = custom_layers.MaskFilter(num_classes=n_class, train_mode=True)([reshaped_filter_layers, y])
        mask_without_y = custom_layers.MaskFilter(num_classes=n_class, train_mode=False)(
            [filter_layers, reshaped_filter_layers, one_hot_layer])

        reshaped_y_train = layers.Reshape((self.input_shape[0] * self.input_shape[1], n_class),
                                          input_shape=(self.input_shape[0], self.input_shape[1], n_class))(y)
        masked_filters_train = K.batch_dot(reshaped_filter_layers, reshaped_y_train)
        masked_filters_train = layers.Flatten()(masked_filters_train)

        # test/prediction condition
        one_hot_layer_reshaped = layers.Reshape((n_class, self.input_shape[0] * self.input_shape[1]),
                                                input_shape=(self.input_shape[0], self.input_shape[1], n_class))(
            one_hot_layer)
        reshaped_y_test = K.one_hot(indices=K.argmax(one_hot_layer_reshaped, 1), num_classes=n_class)

        masked_filters_test = K.batch_dot(reshaped_filter_layers, reshaped_y_test)
        masked_filters_test = layers.Flatten()(masked_filters_test)
        print('masked_filters_test', masked_filters_test)
        print('masked_filters_train', masked_filters_train)
        # Shared Decoder model in training and prediction
        shape_default_mask = (shape_default[0], shape_default[1], n_class)
        decoder = models.Sequential(name='decoder')
        decoder.add(layers.Dense(512, activation='selu', input_dim=(n_filter * n_class)))  # 1024
        decoder.add(layers.Dense(1024, activation='selu'))  # 2048
        decoder.add(layers.Dense(np.prod(shape_default_mask), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=shape_default_mask, name='out_recon'))

        train_model = models.Model([x, y], [one_hot_layer, decoder(mask_with_y)])
        eval_model = models.Model(x, one_hot_layer)

        # eval_model = models.Model(x, [one_hot_layer, decoder(mask_without_y)])
        '''
        train_model = models.Model(x,one_hot_layer)

        eval_model = models.Model(x, one_hot_layer)
        '''
        return train_model, eval_model

    def Run(self, select_model="UNetCapsuleNetClippedModel", parallel=True):
        if (parallel):
            with tf_X.device('/cpu:0'):
                # config = tf_X.ConfigProto()
                # config.gpu_options.per_process_gpu_memory_fraction = 1
                # config.gpu_options.allow_growth = True
                # set_session(tf_X.Session(config=config))
                if select_model=="UNetCapsuleNetClippedModel":
                    model, eval_model = self.UNetCapsuleNetClippedModel(n_class=self.nb_class,
                                                                    n_channels=self.args.n_channels
                                                                    )
                if select_model=="CapsuleNetClippedModel":
                    model, eval_model = self.CapsuleNetClippedModel(n_class=self.nb_class,
                                                                    n_channels=self.args.n_channels,
                                                                    routing=self.args.routings,
                                                                    dim_capsule=self.args.dim_capsule
                                                                    )
                '''
                model, eval_model = self.SimpleNetwerk(n_class=self.args.nclass)
                '''
                # model = eval_model
                model.summary()
        else:
            if select_model == "CapsuleNetClippedModel":
                model, eval_model = self.CapsuleNetClippedModel(n_class=self.nb_class,
                                                                routing=self.args.routings,
                                                                dim_capsule=self.args.dim_capsule,
                                                                n_channels=self.args.n_channels)
            if select_model=="UNetCapsuleNetClippedModel":
                model, eval_model = self.UNetCapsuleNetClippedModel(n_class=self.nb_class,
                                                                    n_channels=self.args.n_channels
                                                                    )
            '''
            model, eval_model = self.SimpleNetwerk(n_class=self.args.nclass)
            '''

        from keras.models import model_from_json, load_model
        # ------------ save the template model rather than the gpu_mode ----------------
        # serialize model to JSON
        if self.args.load_model:
            print(model.summary())
            print("Proc: Loading the previous model...")
            # model = load_model(self.args.save_dir + '/trained_model_g.h5', {'tf': tf, 'iou_loss': self.iou_loss, 'precision': self.precision, 'IOU':self.IOU})
            # load json and create model
            loaded_model_json = None
            with open(self.args.save_dir + '/trained_model.json', "r") as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)

            # loaded_model_json = None
            # with open(self.args.save_dir+'/eval_model.json', 'r') as json_file:
            #    loaded_model_json = json_file.read()
            # eval_model = model_from_json(loaded_model_json)
        # Load the weight is available...
        if self.args.weights is None and self.args.load_previous_weight:
            file_name_weight = self.GetLastWeight(self.args.save_dir)
        else:
            file_name_weight = self.args.weights

        if file_name_weight is not None:
            model.load_weights(file_name_weight)
        else:
            print('No weights are provided. Will test using random initialized weights.')

        # Show the model
        # Testing..
        if not self.args.testing:
            plot_model(model, to_file=self.args.save_dir + '/model.png', show_shapes=True)

            if (parallel):
                print("Parallel:")
                from keras.utils import multi_gpu_model
                multi_model = multi_gpu_model(model, gpus=self.args.gpu)
                plot_model(multi_model, to_file=self.args.save_dir + '/model_parallel.png', show_shapes=True)

                multi_model.summary()
                model = self.train(model=multi_model, multi_gpu=parallel,
                                   load_augmented_data=self.args.load_augmented_data)
            else:
                model = self.train(model=model, multi_gpu=parallel, load_augmented_data=self.args.load_augmented_data)

            file_name_weight = self.GetLastWeight(self.args.save_dir)
            # eval_model = model
            if file_name_weight is not None:
                print("Loading the weight...")
                eval_model.load_weights(file_name_weight, by_name=True)
            self.test(model=eval_model, Onlive=True,
                      image_source="/home/eminaga/EncryptedData/Challenge/MoNuSeg Training Data/Tissue images/")
        else:
            if file_name_weight is not None:
                print("Loading the weight...")
                eval_model.load_weights(file_name_weight, by_name=True)
            if os.path.exists('./scaler.pkl'):
                self.scaler = joblib.load('./scaler.pkl')
                print("Loaded existing scale transformeer...")
            eval_model.summary()

            self.test(model=eval_model, Onlive=True,image_source="/home/eminaga/EncryptedData/Challenge/MoNuSeg Training Data/Tissue images/")