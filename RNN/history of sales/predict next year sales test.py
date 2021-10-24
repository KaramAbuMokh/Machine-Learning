import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # reading the data
    df=pd.read_csv('../DATA/RSCCASN.csv',parse_dates=True,index_col='DATE')
    # parse_dates=True : convert the date strings to date objects

    # info about the data
    # print(df.info())

    # plot the data frame
    # df.plot()
    # plt.show()

    # we can see that the data is monthly
    # so we will take month and a half for the test data
    # that means that the size of the test set is 18
    test_size=18
    test_ind=len(df)-18

    # set the train and test sets
    train=df.iloc[:test_ind]
    test=df.iloc[test_ind:]

    # scaling the data
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    scaler.fit(train)
    scaled_train=scaler.transform(train)
    scaled_test = scaler.transform(test)

    # create the time generator
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    # ******* the batches should be less than the test set

    length=12
    generator=TimeseriesGenerator(scaled_train
                                  ,scaled_train
                                  ,length=length
                                  ,batch_size=1)

    n_features = 1
    '''
    
    
    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,LSTM
    

    model=Sequential()
    model.add(LSTM(100,activation='relu'
                   ,input_shape=(length,n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')

    # adding early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',patience=2)

    # creating the validation generator to validate the test set
    validation_generator=TimeseriesGenerator(scaled_test
                                             ,scaled_test
                                             ,length=length
                                             ,batch_size=1)

    # train the model
    model.fit_generator(generator
                        ,epochs=20
                        ,validation_data=validation_generator
                        ,callbacks=[early_stop])

    # save the model
    model.save('sales model.h5')

    # save the history
    his = pd.DataFrame(model.history.history)
    his.to_csv('history of the model.csv', index=False)
    
    '''

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('sales model.h5')

    # load the history
    models_history = pd.read_csv('history of the model.csv')

    models_history.plot()
    plt.show()

    # organize the test set and predict it
    predictions=[]
    current_batch=scaled_train[-length:]
    current_batch=current_batch.reshape((1,length,n_features))

    for i in range(len(test)):
        curr_pred=my_model.predict(current_batch)[0]
        predictions.append(curr_pred)
        current_batch=np.append(current_batch[:,1:,:]
                                ,[[curr_pred]]
                                ,axis=1)

    # inverse transform
    true_predictions=scaler.inverse_transform(predictions)
    test['Predictions']=true_predictions
    print(test)
    test.plot()
    plt.show()