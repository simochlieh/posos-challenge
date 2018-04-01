from keras import callbacks
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, GRU, Input, BatchNormalization, \
    Convolution1D, MaxPooling1D, Concatenate, Lambda
from keras.optimizers import Adam
from keras.models import Model
from sklearn.utils import class_weight

from feed_func import *
import utils
from tensorboard_callback import MyTensorBoard
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
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


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, parsing_size, max_sentence_length,
                     drop=0.8, regu=None):
    embedding = Lambda(lambda x: x[:, :, :embedding_size, :])(input_)
    e_conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), padding='valid', activation='relu', name='e_conv_0')(
        embedding)
    e_conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), padding='valid', activation='relu', name='e_conv_1')(
        embedding)

    e_maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='e_maxpool_0')(e_conv_0)
    e_maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[1] + 1, 1), strides=(1, 1), name='e_maxpool_1')(e_conv_1)

    parsing = Lambda(lambda x: x[:, :, embedding_size:, :])(input_)

    p_conv_0 = Conv2D(num_filters, (filter_sizes[0], parsing_size), padding='valid', activation='relu',
                      name='p_conv_0')(parsing)
    p_conv_1 = Conv2D(num_filters, (filter_sizes[1], parsing_size), padding='valid', activation='relu',
                      name='p_conv_1')(parsing)

    p_maxpool_0 = MaxPooling2D((max_sentence_length - filter_sizes[0] + 1, 1), strides=(1, 1), name='p_maxpool_0')(
        p_conv_0)
    p_maxpool_1 = MaxPooling2D((max_sentence_length - filter_sizes[1] + 1, 1), strides=(1, 1), name='p_maxpool_1')(
        p_conv_1)

    merged_tensor = concatenate([e_maxpool_0, e_maxpool_1, p_maxpool_0, p_maxpool_1], axis=1)
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
    input_train = np.load(utils.get_X_train_path(args.embedding_dir_path))
    embedding_dim = utils.get_embedding_dim()
    parsing_size = input_train.shape[2] - embedding_dim
    print("Training data has shape: %s" % str(input_train.shape))

    input_train = input_train[:, :, :embedding_dim + parsing_size, np.newaxis]

    y_train = np.load(utils.get_y_train_path(args.embedding_dir_path))
    y_train = to_categorical(y_train, num_classes=len(CLASSES))

    input_test = np.load(utils.get_X_test_path(args.embedding_dir_path))
    input_test = input_test[:, :, :embedding_dim + parsing_size, np.newaxis]

    input_final_test = np.load(utils.get_test_embeddings_path(args.embedding_dir_path))
    input_final_test = input_final_test[:, :, :embedding_dim + parsing_size, np.newaxis]

    y_test = np.load(utils.get_y_test_path(args.embedding_dir_path))
    y_test = to_categorical(y_test, num_classes=len(CLASSES))

    all_input = np.vstack([input_train, input_test])
    all_y = np.vstack([y_train, y_test])

    # Hyper-parameters
    filter_sizes = [1, 2]
    num_filters = 2000
    batch_size = 32
    nb_epochs = 20

    max_sentence_length = max([len(sentence) for sentence in input_train])

    input_ = Input(shape=(max_sentence_length, embedding_dim + parsing_size, 1))
    output = cnn_model_output(input_, num_filters, filter_sizes, embedding_dim, parsing_size,
                              max_sentence_length, regu=args.regularization)
    model = Model(input_, output)
    Training = True
    if Training:
        # Compile
        adam = Adam(lr=1e-3)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', top_5_acc])

        # Prompt user to type in location for the logdir
        # loc = input("Type in specification for log dir name:")
        loc = 'cnn_100_first_tfidf'
        # Callbacks for tensorboard
        logdir = './results/logdir/' + loc + '/'
        tb = MyTensorBoard(log_dir=logdir, histogram_freq=0, write_batch_performance=True)
        # Checkpoint
        # reduceLROnplateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=0.00001)
        checkpointer = ModelCheckpoint(filepath='model_no_corr_50_first_tfidf_all_4.h5', verbose=0, save_best_only=True,
                                       save_weights_only=True, monitor='val_loss')
        model.fit(input_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(input_test, y_test),
                  verbose=2, callbacks=[reduce_lr, checkpointer])
        # model.save_weights('model_no_corr_50_first_tfidf_all_4.h5')
    else:
        model.load_weights('model_no_corr_50_first_tfidf_all_4.h5')
        y_pred = np.argmax(model.predict(input_final_test), axis=1)
        y_test_o = np.argmax(y_test, axis=1)
        acc = metrics.accuracy_score(y_test_o, y_pred)
        print(acc)
        utils.to_csv(y_pred, './results/cnn/y_pred_final_f_4.csv')


if __name__ == '__main__':
    args_ = parser.parse_args()
    main(args_)
