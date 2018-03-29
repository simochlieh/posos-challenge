from keras import callbacks
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization, \
    Convolution1D, MaxPooling1D, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from feed_func import *
import utils
from tensorboard_callback import MyTensorBoard
from keras.callbacks import TensorBoard
import argparse
import numpy as np
from sklearn import metrics
from params import CLASSES
from keras import metrics as kmetrics

parser = argparse.ArgumentParser(description="This script runs the main experiments.")

parser.add_argument('--regularization',
                    default='BNDO',
                    help="Should regularization method between DO and/or BN")

parser.add_argument('--embedding_dir_path',
                    default='./results/embedding/fast_text_embedding_top_50_tfidf_no_corr_w_drug_emb_test_0/',
                    help="Directory storing X_train.py, y_train.py, X_test.py, y_test.py")


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, max_sentence_length, drop=0.8, regu=None):
    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), padding='valid', activation='relu', name='conv_0')(
        input_)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), padding='valid', activation='relu', name='conv_1')(
        input_)
    # conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), padding='valid', activation='relu', name='conv_2')(
    #     input_)
    # conv_3 = Conv2D(num_filters, (filter_sizes[3], embedding_size), padding='valid', activation='relu', name='conv_3')(
    #     input_)
    # conv_4 = Conv2D(num_filters, (filter_sizes[4], embedding_size), padding='valid', activation='relu', name='conv_4')(
    #     input_)

    maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='maxpool_0')(conv_0)
    maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[1] + 1, 1), strides=(1, 1), name='maxpool_1')(conv_1)
    # maxpool_2 = MaxPooling2D((max_sentence_length - filter_sizes[2] + 1, 1), strides=(1, 1), name='maxpool_2')(conv_2)
    # maxpool_3 = MaxPooling2D((max_sentence_length - filter_sizes[3] + 1, 1), strides=(1, 1), name='maxpool_3')(conv_3)
    # maxpool_4 = MaxPooling2D((max_sentence_length - filter_sizes[4] + 1, 1), strides=(1, 1), name='maxpool_4')(conv_4)

    merged_tensor = concatenate([maxpool_0, maxpool_1], axis=1)
    flatten = Flatten(name='flatten')(merged_tensor)
    if regu is not None:
        if 'BN' in regu:
            flatten = BatchNormalization(axis=-1)(flatten)
        if 'DO' in regu:
            flatten = Dropout(drop)(flatten)
    # flatten = Dense(256, activation='relu')(flatten)
    # flatten = Dense(128, activation='relu')(flatten)
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
    if regu is not None:
        if 'BN' in regu:
            flatten = BatchNormalization(axis=-1)(flatten)
        if 'DO' in regu:
            flatten = Dropout(drop)(flatten)
    return Dense(len(CLASSES), activation='softmax', name='predictions')(flatten)


def top_5_acc(y_true, y_predict):
    return kmetrics.top_k_categorical_accuracy(y_true, y_predict, k=5)


def main(args):
    # Loading training data
    embedding_dim = utils.get_embedding_dim()
    input_train = np.load(utils.get_X_train_path(args.embedding_dir_path))
    print(input_train.shape)
    # std_train = input_train.std()
    # mean_train = input_train.mean()

    # input_train = (input_train - mean_train) / std_train
    input_train = input_train[:, :, :embedding_dim, np.newaxis]

    y_train = np.load(utils.get_y_train_path(args.embedding_dir_path))
    y_train = to_categorical(y_train, num_classes=len(CLASSES))

    input_test = np.load(utils.get_X_test_path(args.embedding_dir_path))
    # input_test = (input_test - mean_train) / std_train
    input_test = input_test[:, :, :embedding_dim, np.newaxis]

    input_final_test = np.load(utils.get_test_embeddings_path(args.embedding_dir_path))
    input_final_test = input_final_test[:, :, :embedding_dim, np.newaxis]
    # y_final_test = np.load(utils.get_y_test_embeddings_path(args.embedding_dir_path))
    # y_final_test = to_categorical(y_final_test, num_classes=len(CLASSES))

    y_test = np.load(utils.get_y_test_path(args.embedding_dir_path))
    y_test = to_categorical(y_test, num_classes=len(CLASSES))

    all_input = np.vstack([input_train, input_test])
    all_y = np.vstack([y_train, y_test])

    # Hyper-parameters
    filter_sizes = [1, 2]
    num_filters = 1000
    batch_size = 32
    nb_training_examples = input_train.shape[0]
    nb_test_examples = input_test.shape[0]
    steps_per_epoch = math.ceil(nb_training_examples / batch_size)
    validation_steps = math.ceil(nb_test_examples / batch_size)

    max_sentence_length = max([len(sentence) for sentence in input_train])
    print(max_sentence_length)
    nb_epochs = 50

    input_ = Input(shape=(max_sentence_length, embedding_dim, 1))
    output = cnn_model_output(input_, num_filters, filter_sizes, embedding_dim,
                              max_sentence_length, regu=args.regularization)
    # output = rnn_model_output(input_)

    model = Model(input_, output)

    Training = True
    if Training:
        # Compile
        adam = Adam(lr=1e-4)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', top_5_acc])

        # Prompt user to type in location for the logdir
        # loc = input("Type in specification for log dir name:")
        loc = 'cnn_100_first_tfidf'
        # Callbacks for tensorboard
        logdir = './results/logdir/' + loc + '/'
        tb = MyTensorBoard(log_dir=logdir, histogram_freq=0, write_batch_performance=True)
        # Checkpoint
        # reduceLROnplateau

        # Fit on generator
        # model.fit_generator(
        #     generator=batch_generator(input_data=input_train, y=y_train,
        #                               batch_size=batch_size, max_sent_length=max_sentence_length),
        #     steps_per_epoch=steps_per_epoch,
        #     # callbacks=[tb],
        #     validation_data=batch_generator(input_data=input_test, y=y_test,
        #                                     batch_size=batch_size, max_sent_length=max_sentence_length),
        #     validation_steps=validation_steps,
        #     epochs=nb_epochs,
        #     verbose=1,
        #     use_multiprocessing=False,
        #     workers=1
        # )
        early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        model.fit(input_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(input_test, y_test),
                  verbose=2, callbacks=[tb])
        model.save_weights('model_no_corr_50_first_tfidf_all_2.h5')
    else:
        model.load_weights('model_no_corr_50_first_tfidf_all_2.h5')
        y_pred = np.argmax(model.predict(input_test), axis=1)
        y_test_o = np.argmax(y_test, axis=1)
        acc = metrics.accuracy_score(y_test_o, y_pred)
        print(acc)
        utils.to_csv(y_pred, './results/cnn/y_pred_final.csv')


if __name__ == '__main__':
    args_ = parser.parse_args()
    main(args_)
