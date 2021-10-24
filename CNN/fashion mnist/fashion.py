from tensorflow.keras.datasets import fashion_mnist
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    '''
    # to se the images sizes
    d1, d2 = [], []
    for x in X_train:
        a,b=x.shape
        d1.append(a)
        d2.append(b)
    sns.jointplot(d1,d2)
    plt.show()
    '''

    # -------------------------------------
    plt.imshow(X_train[0])
    plt.show()

    # to see the sets shapes
    print(X_train.shape)
    print(X_test.shape)

    # -------------------------------------

    # reshape the sets
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    # -------------------------------------

    # convert the y_train from numbers to binary
    from tensorflow.keras.utils import to_categorical

    y_cat_train = to_categorical(y_train, num_classes=10)
    y_cat_test = to_categorical(y_test, num_classes=10)
    # that will convert 3 --to-->[0,0,0,1,0,0,0,0,0,0]

    # -------------------------------------

    # see the amount of the unique values of the results
    sns.countplot(y_test)
    plt.show()

    # -------------------------------------

    # scale the values
    X_train = X_train / 255
    X_test = X_test / 255

    # -------------------------------------

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential()

    model.add(Conv2D(128, (5, 5), input_shape=(
        28, 28, 1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), input_shape=(
        28, 28, 1), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), input_shape=(
        28, 28, 1), activation='relu', padding='same'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # to see the sets shapes
    print(X_train[1].shape)
    print(X_test[1].shape)
    # creating an early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # training the model
    model.fit(X_train, y_cat_train, validation_data=(
        X_test, y_cat_test), batch_size=8, epochs=25, callbacks=early_stop)

    # save the model
    model.save('fashion saved model.h5')
    his = pd.DataFrame(model.history.history)
    his.to_csv('history of fashion model.csv')

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('fashion saved model.h5')

    # load the history of the model
    history_of_model = pd.read_csv('history of fashion model.csv')

    print(history_of_model.columns)
    # drop the index column
    history_of_model = history_of_model.drop('Unnamed: 0', axis=1)
    print(history_of_model.columns)

    # show the history of loss and accuracy in graph
    history_of_model.plot()
    plt.show()

    # show the metrics names
    print(my_model.metrics_names)

    # predict the test set
    predictions = my_model.predict_classes(X_test)

    # show report
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictions))
