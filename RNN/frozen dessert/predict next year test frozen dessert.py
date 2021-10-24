import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # read the data
    df = pd.read_csv('../../DATA/Frozen_Dessert_Production.csv',
                     parse_dates=True, index_col='DATE')

    df.columns = ['productions']
    test_length = 24
    train = df.iloc[:-test_length]
    test = df.iloc[-test_length:]

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    # decide the parameters
    length = 18  # the length of the input in order to predict the next point
    batch_size = 1  # how many times we return in each batches
    n_features = 1  # input is x and predict y

    # generatort
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    generator = TimeseriesGenerator(
        scaled_train, scaled_train, length=length, batch_size=batch_size)

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential()
    model.add(LSTM(100, input_shape=(length, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # test generator
    test_generator = TimeseriesGenerator(
        scaled_test, scaled_test, length=length, batch_size=batch_size)

    # adding early stop
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # train the model
    model.fit_generator(generator, epochs=20,
                        validation_data=test_generator, callbacks=[early_stop])

    # save the model
    model.save('frozen dessert test model.h5')

    # save the history
    his = pd.DataFrame(model.history.history)
    his.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('frozen dessert test model.h5')

    # load the history
    models_history = pd.read_csv('history of the model.csv')

    models_history.plot()
    plt.show()

    # predict the test set
    predictions = []
    curr_batch = scaled_train[-length:]
    curr_batch = curr_batch.reshape((1, length, n_features))
    for i in range(test_length):
        curr_pred = my_model.predict(curr_batch)[0]
        predictions.append(curr_pred)
        curr_batch = np.append(curr_batch[:, 1:, :], [[curr_pred]], axis=1)

    true_predictions = scaler.inverse_transform(predictions)
    test['Predictions'] = true_predictions
    print(test)

    test.plot()
    plt.show()

    # calculate the RMSE
    from sklearn.metrics import mean_squared_error
    print(np.sqrt(mean_squared_error(
        test['productions'], test['Predictions'])))
