from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# to split the data to training set and test set we import :
from sklearn.model_selection import train_test_split

# to normalize the data we import :
from sklearn.preprocessing import MinMaxScaler

# importing keras for building the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# reading the data
def readFakeData():
    df = pd.read_csv('../../DATA/fake_reg.csv')
    print(df.head().to_string())
    return df
# readFakeData()

# plot graphs about the data


def plotData(df):
    sns.pairplot(df)
    plt.show()
# plotData(readFakeData())

# get the features


def getFeatures(df):
    X = df[['feature1', 'feature2']].values
    return X

# get the label


def getLabel(df):
    y = df['price'].values
    return y

# split the data


def splitData(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# scaling the data


def scaleData(X_train, X_test):
    scaler = MinMaxScaler()

    # fit is to calculate the parameters to scale the data
    scaler.fit(X_train)

    # here we transform the data using the previous parameters
    X_train = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# creating the model


def createModel():
    # Dense means layer that every neuron
    # connected to all of the others in the next layer
    model = Sequential([Dense(4, activation='relu'),
                        Dense(4, activation='relu'),
                        Dense(4, activation='relu'),
                        Dense(1)])

    '''
    # another way to build the model
    model=Sequential()
    model.add(Dense(4,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1))
    '''
    return model

# predict new data


def pred_new_data(x, y):
    scaler = MinMaxScaler()
    new_data = [[x, y]]
    scaler.fit(getFeatures(readFakeData()))
    new_data = scaler.transform(new_data)
    return new_data


# save the model

def save_my_model(model):
    model.save('fake_price_model.h5')

# load the model


def load_my_model():
    return load_model('fake_price_model.h5')


if __name__ == '__main__':
    # read the data
    df = readFakeData()

    # split the features from the labels in the data set
    X, y = getFeatures(df), getLabel(df)

    # split the training set and the testing set
    X_train, X_test, y_train, y_test = splitData(X, y)

    # scaling the features
    X_train, X_test = scaleData(X_train, X_test)

    # creating the model
    model = createModel()
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x=X_train, y=y_train, epochs=250,)

    # to see the loss history
    loss_df = pd.DataFrame(model.history.history)
    print(loss_df)
    loss_df.plot()
    plt.show()

    # evaluating the model
    print(' the loss of the test set: ')
    print(model.evaluate(X_test, y_test, verbose=0))
    print(' the loss of the train set: ')
    print(model.evaluate(X_train, y_train, verbose=0))

    # predict the test set
    test_predictions = model.predict(X_test)

    # reshape and make the results in a series
    test_predictions = pd.Series(test_predictions.reshape(300,))

    # make the actual results in a data frame
    pred_df = pd.DataFrame(y_test, columns=['Test True Y'])

    # concat the actual results with the predicted results
    pred_df = pd.concat([pred_df, test_predictions], axis=1)

    # rename the columns
    pred_df.columns = ['Test True Y', 'Model Predictions']
    print(pred_df)

    # plot the results
    sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)
    plt.show()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    print('mean_absolute_error : ')
    print(mean_absolute_error(
        pred_df['Test True Y'], pred_df['Model Predictions']))
    print('mean_squared_error : ')
    print(mean_squared_error(
        pred_df['Test True Y'], pred_df['Model Predictions']))

    # predict new data
    new_data = pred_new_data(999, 1000)
    print(model.predict(new_data))

    # save the model
    save_my_model(model)

    # load the model
    my_model = load_my_model()
