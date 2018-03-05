from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from feed_func import *
import utils
from tensorboard_callback import MyTensorBoard
from keras.callbacks import TensorBoard
import argparse
import numpy as np
from params import CLASSES

parser = argparse.ArgumentParser(description="This script runs the main experiments.")

parser.add_argument('--regularization',
                    default=None,
                    help="Should regularization method between DO and/or BN")

parser.add_argument('--embedding_dir_path',
                    default='./results/embedding/fast_text_embedding_wo_stop_words/',
                    help="Directory storing X_train.py, y_train.py, X_tst.py, y_test.py")


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, max_sentence_length, drop=0.8, regu=None):
    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), padding='valid', activation='relu', name='conv_0')(
        input_)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), padding='valid', activation='relu', name='conv_1')(
        input_)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), padding='valid', activation='relu', name='conv_2')(
        input_)

    maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='maxpool_0')(conv_0)
    maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[1] + 1, 1), strides=(1, 1), name='maxpool_1')(conv_1)
    maxpool_2 = MaxPooling2D((max_sentence_length - filter_sizes[2] + 1, 1), strides=(1, 1), name='maxpool_2')(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten(name='flatten')(merged_tensor)
    if regu is not None:
        if 'BN' in regu:
            flatten = BatchNormalization(axis=-1)(flatten)
        if 'DO' in regu:
            flatten = Dropout(drop)(flatten)
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


def main(args):
    # Loading training data
    input_train = np.load(utils.get_X_train_path(args.embedding_dir_path))
    y_train = np.load(utils.get_y_train_path(args.embedding_dir_path))
    input_test = np.load(utils.get_X_test_path(args.embedding_dir_path))
    y_test = np.load(utils.get_y_test_path(args.embedding_dir_path))

    # Hyper-parameters
    filter_sizes = [3, 4, 5]
    num_filters = 8
    batch_size = 50
    nb_training_examples = input_train.shape[0]
    nb_test_examples = input_test.shape[0]
    steps_per_epoch = math.ceil(nb_training_examples / batch_size)
    validation_steps = math.ceil(nb_test_examples / batch_size)
    embedding_dim = utils.get_embedding_dim()
    max_sentence_length = 100
    nb_epochs = 10

    input_ = Input(shape=(max_sentence_length, embedding_dim, 1))
    output = cnn_model_output(input_, num_filters, filter_sizes, embedding_dim,
                              max_sentence_length, regu=args.regularization)
    # output = rnn_model_output(input_)

    model = Model(input_, output)

    # Compile
    adam = Adam(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    # Prompt user to type in location for the logdir
    loc = input("Type in specification for log dir name:")
    # loc = 'final'
    # Callbacks for tensorboard
    logdir = './results/logdir/' + loc + '/'
    tb = MyTensorBoard(log_dir=logdir, histogram_freq=0, write_batch_performance=True)

    # Fit on generator
    model.fit_generator(
        generator=batch_generator(input_data=input_train, y=y_train,
                                  batch_size=batch_size, max_sent_length=max_sentence_length),
        steps_per_epoch=steps_per_epoch,
        # callbacks=[tb],
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
