import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # import the images
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train/255
    X_test = X_test/255

    '''
    
    # make sample of noisy images
    from tensorflow.keras.layers import GaussianNoise
    sample=GaussianNoise(0.2)
    noisy=sample(X_test[:10],training=True)

    # show the difference
    n=0
    plt.imshow(X_test[n])
    plt.show()
    plt.imshow(noisy[n])
    plt.show()
    
    '''

    # make the noisy images
    from tensorflow.keras.layers import GaussianNoise

    sample = GaussianNoise(0.5)
    noisy = sample(X_train, training=True)

    # ----------------------------------------

    # creating the model
    import tensorflow as tf
    from tensorflow.keras.layers import GaussianNoise
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Reshape

    tf.random.set_seed(101)
    np.random.seed(101)

    encoder = Sequential()
    encoder.add(Flatten(input_shape=(28, 28)))
    # encoder.add(GaussianNoise(0.7))
    encoder.add(Dense(400, activation='relu'))
    encoder.add(Dense(200, activation='relu'))
    encoder.add(Dense(100, activation='relu'))
    encoder.add(Dense(50, activation='relu'))
    encoder.add(Dense(25, activation='relu'))

    decoder = Sequential()
    decoder.add(Dense(50, input_shape=[25], activation='relu'))
    decoder.add(Dense(100, activation='relu'))
    decoder.add(Dense(200, activation='relu'))
    decoder.add(Dense(400, activation='relu'))
    decoder.add(Dense(784, activation='sigmoid'))
    decoder.add(Reshape([28, 28]))

    autoencoder = Sequential([encoder, decoder])

    autoencoder.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.fit(noisy, X_train, epochs=8, validation_data=(X_test, X_test))

    # save the model
    autoencoder.save('model to reduce dims.h5')
    # save the encoder
    encoder.save('encoder to reduce dims.h5')
    # save the model
    decoder.save('decoder to reduce dims.h5')

    # ----------------------------------------

    # save the history of the model
    models_history = pd.DataFrame(autoencoder.history.history)
    models_history.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model

    autoencoder = load_model('model to reduce dims.h5')
    encoder = load_model('encoder to reduce dims.h5')
    decoder = load_model('decoder to reduce dims.h5')

    # load the history od the model
    models_history = pd.read_csv('history of the model.csv')

    # plot the validation loss and accuracy
    models_history.plot()
    plt.show()

    # make a test
    sample = GaussianNoise(0.5)

    ten_noisy_images = sample(X_test[:10], training=True)
    denoised = autoencoder(ten_noisy_images)

    n = 0
    plt.imshow(X_test[n])
    plt.show()
    plt.imshow(ten_noisy_images[n])
    plt.show()
    plt.imshow(denoised[n])
    plt.show()
