from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from feed_func import *
import utils
from tensorboard_callback import MyTensorBoard
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import argparse
import numpy as np
from params import CLASSES

parser = argparse.ArgumentParser(description="This script runs the main experiments.")

parser.add_argument('--regularization',
                    default=None,
                    help="Should regularization method between DO and/or BN")

parser.add_argument('--embedding_dir_path',
                    default='./results/embedding/small_fast_text_embedding/',
                    help="Directory storing X_train.py, y_train.py, X_tst.py, y_test.py")


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, max_sentence_length, drop=0.5, regu=None):
    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), padding='valid', activation='relu', name='conv_0')(
        input_)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), padding='valid', activation='relu', name='conv_1')(
        input_)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), padding='valid', activation='relu', name='conv_2')(
        input_)
    conv_3 = Conv2D(num_filters, (filter_sizes[3], embedding_size), padding='valid', activation='relu', name='conv_3')(
        input_)
    conv_4 = Conv2D(num_filters, (filter_sizes[4], embedding_size), padding='valid', activation='relu', name='conv_4')(
        input_)

    maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='maxpool_0')(conv_0)
    maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[1] + 1, 1), strides=(1, 1), name='maxpool_1')(conv_1)
    maxpool_2 = MaxPooling2D((max_sentence_length - filter_sizes[2] + 1, 1), strides=(1, 1), name='maxpool_2')(conv_2)
    maxpool_3 = MaxPooling2D((max_sentence_length - filter_sizes[3] + 1, 1), strides=(1, 1), name='maxpool_3')(conv_3)
    maxpool_4 = MaxPooling2D((max_sentence_length - filter_sizes[4] + 1, 1), strides=(1, 1), name='maxpool_4')(conv_4)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4], axis=1)
    flatten = Flatten(name='flatten')(merged_tensor)

    if regu is not None:
        if 'BN' in regu:
            flatten = BatchNormalization(axis=-1)(flatten)
        if 'DO' in regu:
            flatten = Dropout(drop)(flatten)

    # dense = Dense(120, activation='softmax', name='dense')(flatten)
    # if regu is not None:
    #     if 'BN' in regu:
    #         dense = BatchNormalization(axis=-1)(dense)
    #     if 'DO' in regu:
    #         dense = Dropout(drop)(dense)
    
    return Dense(len(CLASSES), activation='softmax', name='predictions')(flatten)


def rnn_model_output(input_, drop=0.8, regu=None):
    # input_ = Reshape((BATCH_SIZE, get_max_sent_length(), get_embedding_dim()))(input_)
    rec = GRU(150)(input_)
    dense1 = Dense(80)(rec)
    if regu is not None:
        if 'BN' in regu:
            dense1 = BatchNormalization(axis=-1)(dense1)
        if 'DO' in regu:
            dense1 = Dropout(drop)(dense1)
    dense2 = Dense(len(CLASSES), activation='softmax', name='predictions')(dense1)

    return dense2



def rm_cnn(input_, num_filters, filter_sizes, embedding_size, max_sentence_length, drop=0.2, regu=None):
    conv_01 = Conv2D(num_filters[0], (filter_sizes[0][0], embedding_size), padding='valid', activation='relu',
                     name='conv_01')(
        input_)
    conv_11 = Conv2D(num_filters[0], (filter_sizes[0][1], embedding_size), padding='valid', activation='relu',
                     name='conv_11')(
        input_)
    conv_21 = Conv2D(num_filters[0], (filter_sizes[0][2], embedding_size), padding='valid', activation='relu',
                     name='conv_21')(
        input_)

    maxpool_01 = MaxPooling2D((2, 1), name='maxpool_01')(conv_01)
    maxpool_11 = MaxPooling2D((2, 1), name='maxpool_11')(conv_11)
    maxpool_21 = MaxPooling2D((2, 1), name='maxpool_21')(conv_21)

    conv_02 = Conv2D(num_filters[1], (filter_sizes[1][0], 1), padding='valid', activation='relu',
                     name='conv_02')(
        maxpool_01)
    conv_12 = Conv2D(num_filters[1], (filter_sizes[1][1], 1), padding='valid', activation='relu',
                     name='conv_12')(
        maxpool_11)
    conv_22 = Conv2D(num_filters[1], (filter_sizes[1][2], 1), padding='valid', activation='relu',
                     name='conv_22')(
        maxpool_21)

    maxpool_02 = MaxPooling2D((2, 1), name='maxpool_02')(conv_02)
    maxpool_12 = MaxPooling2D((2, 1), name='maxpool_12')(conv_12)
    maxpool_22 = MaxPooling2D((2, 1), name='maxpool_22')(conv_22)

    merged_tensor = concatenate([maxpool_02, maxpool_12, maxpool_22], axis=1)
    flatten = Flatten(name='flatten')(merged_tensor)
    if regu is not None:
        if 'BN' in regu:
            flatten = BatchNormalization(axis=-1)(flatten)
        if 'DO' in regu:
            flatten = Dropout(drop)(flatten)
    return Dense(len(CLASSES), activation='softmax', name='predictions')(flatten)


def main(args):
    # Loading training data
    input_train = np.load(utils.get_X_train_path(args.embedding_dir_path))
    y_train = np.load(utils.get_y_train_path(args.embedding_dir_path))
    input_test = np.load(utils.get_X_test_path(args.embedding_dir_path))
    y_test = np.load(utils.get_y_test_path(args.embedding_dir_path))

    # Hyper-parameters
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 25
    batch_size = 50
    nb_training_examples = input_train.shape[0]
    nb_test_examples = input_test.shape[0]
    steps_per_epoch = math.ceil(nb_training_examples / batch_size)
    validation_steps = math.ceil(nb_test_examples / batch_size)
    embedding_dim = utils.get_embedding_dim()
    max_sentence_length = 100
    nb_epochs = 20

    input_ = Input(shape=(max_sentence_length, embedding_dim, 1))
    output = cnn_model_output(input_, num_filters, filter_sizes, embedding_dim,
                              max_sentence_length, regu=args.regularization)
    # output = rnn_model_output(input_)

    model = Model(input_, output)

    # Compile
    adam = Adam(lr=1e-2)
    # Prompt user to type in location for the logdir
    loc = input("Type in specification for log dir name:")
    # loc = 'final'
    # Callbacks for tensorboard
    logdir = './results/logdir/' + loc + '/'
    try:
        model.load_weights('%smodel-ckpt'%logdir)
        print('Existing model found, taking over existing weights.')
    except OSError:
        print('No existing model found. Starting training from scratch.')

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    tb = MyTensorBoard(log_dir=logdir, histogram_freq=0, write_batch_performance=True)
    # Checkpoint
    checkpointer = ModelCheckpoint(filepath='%smodel-ckpt'%logdir, verbose=0, save_best_only=True)
    # reduceLROnplateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.001)

    # Fit on generator
    model.fit_generator(
        generator=batch_generator(input_data=input_train, y=y_train,
                                  batch_size=batch_size, max_sent_length=max_sentence_length),
        steps_per_epoch=steps_per_epoch,
        callbacks=[tb,checkpointer, reduce_lr],
        validation_data=batch_generator(input_data=input_test, y=y_test,
                                        batch_size=batch_size, max_sent_length=max_sentence_length),
        validation_steps=validation_steps,
        epochs=nb_epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=1
    )


if __name__ == '__main__':
    args_ = parser.parse_args()
    main(args_)
