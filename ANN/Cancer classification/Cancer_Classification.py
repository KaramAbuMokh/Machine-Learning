import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# read the data
def read_cancer_file():
    df = pd.read_csv('../../DATA/cancer_classification.csv')
    print(df.head().to_string())
    return df


# some info about the data
def informations(df):
    print(df.info())
# informations(read_cancer_file())


# describtion about the data
def desc(df):
    print(df.describe().transpose().to_string())
# desc(read_cancer_file())

# count the unique values and put them in a graph


def count_values(df):
    sns.countplot(x='benign_0__mal_1', data=df)
    plt.show()
# count_values(read_cancer_file())

# see the correlations between the data


def corr_data(df):
    print(df.corr().to_string())
    return df.corr()
# corr_data(read_cancer_file())


# correlation of the label we want to
# predict with the other labels
def corr_of_nenign_mal():
    correlation_df = corr_data(read_cancer_file())
    print(correlation_df['benign_0__mal_1'].sort_values())
    return correlation_df['benign_0__mal_1'].sort_values()
# corr_of_nenign_mal()

# plot the correlation of the benign mal label


def plot_corr_of_nenign_mal(df):
    df.plot(kind='bar')
    plt.show()
# plot_corr_of_nenign_mal(corr_of_nenign_mal())


if __name__ == '__main__':

    # read the data
    df = read_cancer_file()

    # split the data from the result
    X = df.drop('benign_0__mal_1', axis=1).values
    y = df['benign_0__mal_1'].values

    # split the test and the train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential()
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.5))
    # the dropp is to turn off 0.5 of the neurons randomly in every epoch

    # binary classification problem ( sigmoid)
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    # model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test))

    # the early stop
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=25)
    # explaine :
    #           patience : wait 25 more epochs before stopping
    #           verbose:   return a report
    #           mode:   minimize the val_loss

    model.fit(x=X_train, y=y_train, epochs=600,
              validation_data=(X_test, y_test),
              callbacks=[early_stop])

    # save the model and the history
    model.save('cancer_model.h5')
    history_df = pd.DataFrame(model.history.history)
    history_df.to_csv('history of cancer model.csv', index=False)

    # load the saved model
    from tensorflow.keras.models import load_model
    my_model = load_model('cancer_model.h5')

    # load the history
    history_of_model = pd.read_csv('history of cancer model.csv')
    print(history_of_model)
    history_of_model.plot()
    plt.show()

    # predict the test set
    predictions = my_model.predict_classes(X_test)

    # reports
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_test, predictions))
    # the printed shows that the model missed ones
    print(confusion_matrix(y_test, predictions))
