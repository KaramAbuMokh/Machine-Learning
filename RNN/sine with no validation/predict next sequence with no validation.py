import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # generate the set
    x = np.linspace(0, 50, 501)
    y = np.sin(x)

    # to data frame
    df = pd.DataFrame(data=y, index=x, columns=['sine'])

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaler.fit(df)
    scaled_set = scaler.transform(df)

    # creating the generator
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    generator = TimeseriesGenerator(
        scaled_set, scaled_set, length=50, batch_size=1)

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense
    model = Sequential()
    model.add(SimpleRNN(50, input_shape=(50, 1)))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit_generator(generator, epochs=5)

    # save the model
    model.save('sequence sine model.h5')

    # save the history
    his = pd.DataFrame(model.history.history)
    his.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('sequence sine model.h5')

    # load the history
    models_history = pd.read_csv('history of the model.csv')

    forcast = []
    first_eval_batch = scaled_set[-50:]
    current_batch = first_eval_batch.reshape((1, 50, 1))

    for i in range(25):
        curr_pred = my_model.predict(current_batch)[0]
        forcast.append(curr_pred)
        current_batch = np.append(
            current_batch[:, 1:, ], [[curr_pred]], axis=1)

    # inverse transform
    forcast = scaler.inverse_transform(forcast)

    # create the indexes
    indexes = np.arange(50.1, 52.6, step=0.1)

    plt.plot(df.index, df['sine'])
    plt.plot(indexes, forcast)
    plt.show()
    print(forcast)
