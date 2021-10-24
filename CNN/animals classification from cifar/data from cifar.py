import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import cifar10

# importing the data


def read_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':

    # read the data
    (X_train, y_train), (X_test, y_test) = read_data()

    # scale the data
    X_test = X_test/255
    X_train = X_train/255

    # convert the y_train and test from numbers to binary
    # that will convert 3 --to-->[0,0,0,1,0,0,0,0,0,0]
    from tensorflow.keras.utils import to_categorical
    y_cat_train = to_categorical(y_train)
    y_cat_test = to_categorical(y_test)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3),
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3),
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # see the summary of the model structure
    print(model.summary())

    # adding early stop
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # train the model
    model.fit(X_train, y_cat_train, epochs=15, validation_data=(
        X_test, y_cat_test), callbacks=[early_stop])

    # save the model
    model.save('model to predict animals.h5')

    # save the history of the model
    models_history = pd.DataFrame(model.history.history)
    models_history.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('model to predict animals.h5')

    # load the history od the model
    models_history = pd.read_csv('history of the model.csv')

    # plot the validation loss and accuracy
    models_history.plot()
    plt.show()

    # printing the history of the model :loss and accuracy
    print(models_history)

    # plot only the accuracy
    models_history[['accuracy', 'val_accuracy']].plot()
    plt.show()

    # plot only the validation loss
    models_history[['loss', 'val_loss']].plot()
    plt.show()

    # get the predictions
    predictions = my_model.predict_classes(X_test)

    # some evaluation

    # validation loss and the validation accuracy
    print(my_model.evaluate(X_test, y_cat_test, verbose=0))

    from sklearn.metrics import classification_report, confusion_matrix

    # report
    print(classification_report(y_test, predictions))

    # confusion_matrix
    print(confusion_matrix(y_test, predictions))

    # heat map for the confusion_matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True)
    plt.show()
    # we can see that the model make
    # mistakes between the dogs and
    # the cats  (3 and 5)

    # predict image
    my_image = X_test[0]
    plt.imshow(my_image)
    plt.show()
    print(y_test[0])
    print(my_model.predict_classes(my_image.reshape(1, 32, 32, 3)))
