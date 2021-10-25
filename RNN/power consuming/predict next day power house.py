import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # read the data
    df = pd.read_csv('../../DATA/energydata_complete.csv',
                     infer_datetime_format=True, index_col='date')

    # see the data's info to be sure that there is no null data
    # print(df.info())

    # grab the last month data
    df = df.loc['2016-05-01':]

    # round to 2 digits after the poin
    df = df.round(2)

    # set variables
    test_days = 2
    # 10 is because the measures are every 10 minutes
    test_set_length = int(2*24*60/10)
    # one day of data in order to predict the next 10 minutes
    length = int(24*60/10)
    batch_size = 1
    n_features = 28   # number of features to predict

    # set the train and test sets
    train = df.iloc[:-test_set_length]
    test = df.iloc[-test_set_length:]

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    # create the generator
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    generator = TimeseriesGenerator(
        scaled_train, scaled_train, length=length, batch_size=batch_size)

    # create the test generator
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    test_generator = TimeseriesGenerator(
        scaled_test, scaled_test, length=length, batch_size=batch_size)

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential()
    model.add(LSTM(100, input_shape=(length, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')

    # adding early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # train the model
    model.fit_generator(generator, epochs=20,
                        validation_data=test_generator, callbacks=[early_stop])

    # save the model
    model.save('power model.h5')

    # save the history
    his = pd.DataFrame(model.history.history)
    his.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('power model.h5')

    # load the history
    models_history = pd.read_csv('history of the model.csv')

    models_history.plot()
    plt.show()

    # predict the test set
    predictions = []
    curr_batch = scaled_train[-length:]
    curr_batch = curr_batch.reshape((1, length, n_features))

    for i in range(len(test)):
        curr_pred = my_model.predict(curr_batch)[0]
        predictions.append(curr_pred)
        curr_batch = np.append(curr_batch[:, 1:, :], [[curr_pred]], axis=1)

    # inverse transformation
    true_predictions = scaler.inverse_transform(predictions)

    # arrange the predictions in a data frame
    true_predictions = pd.DataFrame(
        data=true_predictions, columns=test.columns)

    test['Visibility'].plot()
    true_predictions['Visibility'].plot()
    plt.show()
    print(test.columns)
