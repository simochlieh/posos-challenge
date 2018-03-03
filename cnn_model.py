from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from utils import get_embedding_dim, get_max_sent_length
from keras.models import Model
from feed_func import *
from params import BATCH_SIZE, keras_fit_params, CLASSES
from keras.callbacks import TensorBoard


def cnn_model_output(input_, num_filters, filter_sizes, embedding_size, decision_len, drop):
    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_size), padding='valid', activation='relu', name='conv_0')(
        input_)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_size), padding='valid', activation='relu', name='conv_1')(
        input_)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_size), padding='valid', activation='relu', name='conv_2')(
        input_)

    maxpool_0 = MaxPooling2D((decision_len - filter_sizes[0] + 1, 1), strides=(1, 1), name='maxpool_0')(conv_0)
    maxpool_1 = MaxPooling2D((decision_len - filter_sizes[1] + 1, 1), strides=(1, 1), name='maxpool_1')(conv_1)
    maxpool_2 = MaxPooling2D((decision_len - filter_sizes[2] + 1, 1), strides=(1, 1), name='maxpool_2')(conv_2)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten(name='flatten')(merged_tensor)
    # dropout = Dropout(drop)(flatten)
    return Dense(len(CLASSES), activation='softmax', name='predictions')(flatten)


def main():
    # Hyper-parameters
    filter_sizes = [3, 4, 5]
    num_filters = 8
    drop = 0.8

    input_ = Input(shape=(get_max_sent_length(), get_embedding_dim(), 1))

    output = cnn_model_output(input_, num_filters, filter_sizes, get_embedding_dim(), get_max_sent_length(), drop)
    model = Model(input_, output)

    # Compile
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    # Callbacks for tensorboard
    tb = TensorBoard(log_dir='./logdir', histogram_freq=1)
    l = numpy.load(EMBEDDING_FILEPATH)
    X = list(map(lambda s: numpy.stack(s[:100]), l[:-500]))
    reshaped = map(lambda m: numpy.vstack((m, numpy.zeros((100 - m.shape[0], 300)))).reshape(100, 300, 1), X)
    x = numpy.stack(reshaped)
    Y = pandas.read_csv(get_labels_path(), sep=';').intention.values[:-500]
    y = to_categorical(Y, num_classes=51)
    # score = model.evaluate(x, y, batch_size=50)
    # print(score)

    # Fit on generator
    model.fit_generator(
        generator=batch_generator(),
        callbacks=[tb],
        validation_data=(x, y),
        **keras_fit_params)




if __name__ == '__main__':
    main()
