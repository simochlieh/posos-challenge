from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization, \
    Lambda, Reshape, Convolution1D, MaxPooling1D, Concatenate
from keras.optimizers import Adam
from keras.models import Model, load_model
from feed_func import *
import utils
from tensorboard_callback import MyTensorBoard
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import argparse
import numpy as np
from params import CLASSES
import os
from sklearn.utils import class_weight, shuffle

parser = argparse.ArgumentParser(description="This script runs the main experiments.")

parser.add_argument('--regularization',
                    default='BN',
                    help="Should regularization method between DO and/or BN")

parser.add_argument('--embedding_dir_path',
                    default='./results/embedding/fast_text_embedding_top_100_tfidf_no_corr/',
                    help="Directory storing X_train.py, y_train.py, X_tst.py, y_test.py")

parser.add_argument('--length_bounds',
                    default=None,
                    type=int,
                    help="Filter out sentences longer than that for training (test set is not altered).")


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, parsing_size, max_sentence_length, drop=0.3,
                     regu=None):
    embedding = Lambda(lambda x: x[:, :, :embedding_size, :])(input_)

    e_conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), padding='valid', activation='relu',
                      name='e_conv_0')(
        embedding)
    e_conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), padding='valid', activation='relu',
                      name='e_conv_1')(
        embedding)
    # e_conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), padding='valid', activation='relu',
    #                  name='e_conv_2')(
    #     embedding)
    # e_conv_3 = Conv2D(num_filters, (filter_sizes[3], embedding_size), padding='valid', activation='relu',
    #                   name='e_conv_3')(
    #     embedding)
    # e_conv_4 = Conv2D(num_filters, (filter_sizes[4], embedding_size), padding='valid', activation='relu',
    #                   name='e_conv_4')(
    #     embedding)

    e_maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='e_maxpool_0')(
        e_conv_0)
    e_maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[1] + 1, 1), strides=(1, 1), name='e_maxpool_1')(
        e_conv_1)
    # e_maxpool_2 = MaxPooling2D((max_sentence_length - filter_sizes[2] + 1, 1), strides=(1, 1), name='e_maxpool_2')(
    #     e_conv_2)
    # e_maxpool_3 = MaxPooling2D((max_sentence_length - filter_sizes[3] + 1, 1), strides=(1, 1), name='e_maxpool_3')(
    #     e_conv_3)
    # e_maxpool_4 = MaxPooling2D((max_sentence_length - filter_sizes[4] + 1, 1), strides=(1, 1), name='e_maxpool_4')(
    #     e_conv_4)

    parsing = Lambda(lambda x: x[:, :, embedding_size:, :])(input_)
    # rec = GRU(15)(parsing)
    # reshape_out = Reshape((1, 1, 15))(rec)
    p_conv_0 = Conv2D(5, (filter_sizes[0], parsing_size), padding='valid', activation='relu',
                      name='p_conv_0')(
        parsing)
    p_conv_1 = Conv2D(5, (filter_sizes[2], parsing_size), padding='valid', activation='relu',
                      name='p_conv_1')(
        parsing)
    # p_conv_2 = Conv2D(5, (filter_sizes[2], parsing_size), padding='valid', activation='relu',
    #                   name='p_conv_2')(
    #     parsing)
    # p_conv_3 = Conv2D(5, (filter_sizes[3], parsing_size), padding='valid', activation='relu',
    #                   name='p_conv_3')(
    #     parsing)
    # p_conv_4 = Conv2D(5, (filter_sizes[4], parsing_size), padding='valid', activation='relu',
    #                   name='p_conv_4')(
    #     parsing)

    p_maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='p_maxpool_0')(
        p_conv_0)
    p_maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[2] + 1, 1), strides=(1, 1), name='p_maxpool_1')(
        p_conv_1)
    # p_maxpool_2 = MaxPooling2D((max_sentence_length - filter_sizes[2] + 1, 1), strides=(1, 1), name='p_maxpool_2')(
    #     p_conv_2)
    # p_maxpool_3 = MaxPooling2D((max_sentence_length - filter_sizes[3] + 1, 1), strides=(1, 1), name='p_maxpool_3')(
    #     p_conv_3)
    # p_maxpool_4 = MaxPooling2D((max_sentence_length - filter_sizes[4] + 1, 1), strides=(1, 1), name='p_maxpool_4')(
    #     p_conv_4)

    merged_tensor = concatenate(
        [e_maxpool_0, e_maxpool_1, p_maxpool_0, p_maxpool_1], axis=3)

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


def rm_cnn(input_, num_filters, filter_sizes, embedding_size, max_sentence_length, drop=0.6, regu=None):
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
    print(flatten.shape)
    if regu is not None:
        if 'BN' in regu:
            flatten = BatchNormalization(axis=-1)(flatten)
        if 'DO' in regu:
            flatten = Dropout(drop)(flatten)
    return Dense(len(CLASSES), activation='softmax', name='predictions')(flatten)


def main(args):
    # Loading training data
    input_train = np.load(utils.get_X_train_path(args.embedding_dir_path))
    # print(input_train[0])
    # std_train = input_train.std()
    # mean_train = input_train.mean()

    # input_train = (input_train - mean_train) / std_train
    # input_train = input_train[:, :, :, np.newaxis]

    y_train = np.load(utils.get_y_train_path(args.embedding_dir_path))
    # # Augment by "rotating":
    # input_train, y_train = shuffle(
    #     numpy.concatenate((input_train, numpy.array([s[::-1] for s in input_train]))),
    #     numpy.concatenate((y_train, y_train))
    # )
    # # RUS
    # over_repr = [42, 32, 14, 48, 34, 22, 44, 31, 28]  # More than 200 individuals
    # input_train, y_train = tuple(map(np.array, zip(
    #     *filter(lambda t: (numpy.random.randint(2) if t[1] in over_repr else 1) == 1, zip(input_train, y_train)))))
    # if args.length_bounds is not None:
    #     Xy_train = filter(lambda t: len(t[0]) <= args.length_bounds, zip(input_train, y_train))
    #     input_train, y_train = tuple(map(np.array, zip(*Xy_train)))

    # Get class weights:
    cl_w = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    y_train = to_categorical(y_train, num_classes=len(CLASSES))

    input_test = np.load(utils.get_X_test_path(args.embedding_dir_path))
    # input_test = (input_test - mean_train) / std_train
    # input_test = input_test[:, :, :, np.newaxis]

    y_test = np.load(utils.get_y_test_path(args.embedding_dir_path))
    y_test = to_categorical(y_test, num_classes=len(CLASSES))

    # Hyper-parameters
    filter_sizes = [1, 2, 3]
    num_filters = 135
    batch_size = 50

    nb_training_examples = input_train.shape[0]
    nb_test_examples = input_test.shape[0]
    steps_per_epoch = math.ceil(nb_training_examples / batch_size)
    validation_steps = math.ceil(nb_test_examples / batch_size)
    embedding_dim = utils.get_embedding_dim()
    parsing_dim = utils.get_parsing_dim()
    max_sentence_length = max(map(lambda s: len(s), input_train))
    nb_epochs = 30

    input_ = Input(shape=(max_sentence_length, embedding_dim + parsing_dim, 1))
    output = cnn_model_output(input_, num_filters, filter_sizes, embedding_dim, parsing_dim,
                              max_sentence_length, regu=args.regularization)

    model = Model(input_, output)
    # Compile
    adam = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    # Prompt user to type in location for the logdir
    loc = input("Type in specification for log dir name:")
    # loc = 'final'
    # Callbacks for tensorboard
    logdir = './results/logdir/' + loc + '/'
    try:
        model.load_weights('%smodel-ckpt' % logdir)
        print('Existing model found, taking over existing weights.')
    except OSError:
        print('No existing model found. Starting training from scratch.')
        os.mkdir(logdir)

    # with open(logdir+'model.pkl', 'wb+') as f:
    #     pickle.dump(model, f)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    model.save(logdir + 'model.h5')

    tb = MyTensorBoard(log_dir=logdir, histogram_freq=0, write_batch_performance=True)
    # Checkpoint
    checkpointer = ModelCheckpoint(filepath='%smodel-ckpt' % logdir, verbose=0, save_best_only=True)
    # reduceLROnplateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                  patience=2, min_lr=0.0001)

    # Fit on generator
    model.fit_generator(
        generator=batch_generator(input_data=input_train, y=y_train,
                                  batch_size=batch_size, max_sent_length=108),
        steps_per_epoch=steps_per_epoch,
        callbacks=[tb, checkpointer, reduce_lr],
        validation_data=batch_generator(input_data=input_test, y=y_test,
                                        batch_size=batch_size, max_sent_length=108),
        validation_steps=validation_steps,
        epochs=nb_epochs,
        verbose=1,
        use_multiprocessing=False,
        class_weight=cl_w,
        workers=1
    )


if __name__ == '__main__':
    args_ = parser.parse_args()
    main(args_)
