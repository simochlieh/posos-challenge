import math
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.optimizers import SGD, Adam
from keras.models import Model
from feed_func import *
from params import keras_fit_params, CLASSES, TEST_STEPS_PER_EPOCH
from keras.callbacks import TensorBoard
import utils
from tensorboard_callback import MyTensorBoard

EMBEDDING_DIRPATH = './results/embedding/fast_text_embedding_wo_stop_words/'


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, max_sentence_length):
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
    # dropout = Dropout(drop)(flatten)
    return Dense(len(CLASSES), activation='softmax', name='predictions')(flatten)


def main():
    # Loading training data
    input_train = np.load(utils.get_X_train_path(EMBEDDING_DIRPATH))
    y_train = np.load(utils.get_y_train_path(EMBEDDING_DIRPATH))
    input_test = np.load(utils.get_X_test_path(EMBEDDING_DIRPATH))
    y_test = np.load(utils.get_y_test_path(EMBEDDING_DIRPATH))

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

    output = cnn_model_output(input_, num_filters, filter_sizes, utils.get_embedding_dim(),
                              utils.get_max_sent_length())
    model = Model(input_, output)

    # Compile
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

    # Callbacks for tensorboard
    tb = MyTensorBoard(log_dir='./logdir', histogram_freq=1, write_batch_performance=True)

    # Fit on generator
    model.fit_generator(
        generator=batch_generator(input_data=input_train, y=y_train,
                                  batch_size=batch_size, max_sent_length=max_sentence_length),
        steps_per_epoch=steps_per_epoch,
        callbacks=[tb],
        validation_data=batch_generator(input_data=input_test, y=y_test,
                                        batch_size=batch_size, max_sent_length=max_sentence_length),
        validation_steps=validation_steps,
        epochs=nb_epochs,
        verbose=2,
        use_multiprocessing=False,
        workers=1
    )


if __name__ == '__main__':
    main()
