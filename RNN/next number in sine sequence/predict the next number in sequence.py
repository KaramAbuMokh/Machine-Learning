import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # creating a sequence between 0 and 50
    # with 0.1 difference
    x=np.linspace(0,50,501)

    # create array with the sin of x
    y=np.sin(x)

    # show the sin in graph
    plt.plot(x,y)
    plt.show()

    # creating data frame with the sine and x
    df=pd.DataFrame(data=y,index=x,columns=['sine'])

    # the percent what we want to predict
    test_percent=0.1

    # round to the test set length to real number : 50
    test_point=np.round(len(df)*test_percent)

    # the index of the test set : 451
    test_ind=int(len(df)-test_point)

    # set the train set
    train=df.iloc[:test_ind]

    # set the test set
    test=df.iloc[test_ind:]

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()

    scaler.fit(train)

    scaled_train=scaler.transform(train)
    scaled_test=scaler.transform(test)

    # creating the history using the time
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    # decide the parameters
    length=49    # the length of the input in order to predict the next point
    batch_size=1 # how many times we return in each batches
    n_features = 1  # input is x and predict y

    # creating the generator
    generator=TimeseriesGenerator(scaled_train
                                  ,scaled_train
                                  ,length=length
                                  ,batch_size=batch_size)

    # early stop
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',patience=1)

    # generator for the early stop
    val_generator = TimeseriesGenerator(scaled_test
                                    , scaled_test
                                    , length=length
                                    , batch_size=batch_size)



    
    
    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,SimpleRNN,LSTM



    model=Sequential()
    #model.add(SimpleRNN(49,input_shape=(length,n_features)))
    model.add(LSTM(49,input_shape=(length,n_features)))
    model.add(Dense(10))
    model.add(Dense(2))
    model.add(Dense(1))
    # mse for continues values
    model.compile(optimizer='adam',loss='mse')
    # model.fit_generator(generator,epochs=5)
    model.fit_generator(generator, epochs=20
                        ,validation_data=val_generator
                        ,callbacks=[early_stop])

    # save the model
    model.save('sequence sine model.h5')

    # save the history
    his=pd.DataFrame(model.history.history)
    his.to_csv('history of the model.csv',index=False)




    # load the model
    from tensorflow.keras.models import load_model
    my_model=load_model('sequence sine model.h5')

    # load the history
    models_history=pd.read_csv('history of the model.csv')

    models_history.plot()
    plt.show()

    # predict new value
    first_eval_batch=scaled_train[-length:]
    first_eval_batch=first_eval_batch.reshape((1,length,n_features))
    print(my_model.predict(first_eval_batch))
    print(scaled_test[0])

    # creating the continuing batches
    test_predictions=[]
    first_eval_batch = scaled_train[-length:]
    current_patch = first_eval_batch.reshape((1, length, n_features))

    for i in range(len(test)):
        current_pred=my_model.predict(current_patch)[0]
        test_predictions.append(current_pred)
        current_patch=np.append(current_patch[:,1:,:],[[current_pred]],axis=1)

    # now we have the predictions pf the test
    # set in the variable test_predictions

    # inverse transform of the test predictions
    true_predictions=scaler.inverse_transform(test_predictions)

    # add the predictions to the test set
    test['predictions']=true_predictions
    print(test)

    # plot the test set
    test.plot()
    plt.show()



