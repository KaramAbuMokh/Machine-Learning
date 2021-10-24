import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the data


def read_houses():
    df = pd.read_csv('../../DATA/kc_house_data.csv')
    print(df.head().to_string())
    return df

# for every column sum how many missing data
# there are


def show_missing_data(df):
    data_is_null = df.isnull().sum()
    print(data_is_null)
    return data_is_null

# show distribution of one label


def show_one_label_dist(df, label):
    sns.distplot(df[label], kde=False)
    plt.show()
# show_one_label_dist(read_houses(),'price')

# count every unique value in some label


def count_unique(df, label):
    sns.countplot(df[label])
    plt.show()
# count_unique(read_houses(),'bedrooms')

# correlations label with all the other labels


def cor_label(df, label):
    return df.corr()[label].sort_values()

# show scatter in graph


def show_cor_in_graph(df, labelx, labely):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=labelx, y=labely, data=df)
    plt.show()
# show_cor_in_graph(read_houses(),'long','lat')

# show scatter in graph comparing to price


def show_cor_in_graph_compare_to_price(df, labelx, labely):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=labelx, y=labely, data=df, hue='price')
    plt.show()
# show_cor_in_graph_compare_to_price(read_houses(),'long','lat')

# removing extremely high prices
# edgecolor is the color of the border of the points


def remove_high_prices(df):
    sns.scatterplot(x='long', y='lat', data=df, hue='price')
    plt.show()
    df = df.sort_values('price', ascending=False)
    non_top_1_perc = df.iloc[300:]
    sns.scatterplot(x='long', y='lat', data=non_top_1_perc,
                    edgecolor=None, alpha=0.1, palette='RdYlGn', hue='price')
    plt.show()
    return non_top_1_perc

# drop the id column


def id_drop(df):
    return df.drop('id', axis=1)

# convert the date to type date


def convert_date(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

# extract the year


def extract_year(df):
    df['year'] = df['date'].apply(lambda date: date.year)
    return df

# extract the month


def extract_month(df):
    df['month'] = df['date'].apply(lambda date: date.month)
    return df


if __name__ == '__main__':

    # drop the high prices so the distribution be good
    df = remove_high_prices(read_houses())

    # drop the id column because the id dont affect the price
    df = id_drop(df)

    # drop the zip_code column because the id dont affect the price
    df = df.drop('zipcode', axis=1)

    # convert the date to type date
    df = convert_date(df)

    # extract the year
    df = extract_year(df)

    # extract the month
    df = extract_month(df)

    # drop the date
    df = df.drop('date', axis=1)

    # separating the features from the labels
    X = df.drop('price', axis=1).values    # the features
    y = df['price'].values                # the labels

    # importing library to split to train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # import the library to create the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # creating the model

    model = Sequential()

    model.add(Dense(19, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.add(Dense(19, activation='relu'))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(x=X_train, y=y_train,
              validation_data=(X_test, y_test),
              batch_size=128, epochs=400)

    # save the history of the model in the csv file
    df_history = pd.DataFrame(model.history.history)
    df_history.to_csv('models_history.csv', index=False)

    from tensorflow.keras.models import load_model

    # model.save('house_price_model.h5')
    df_history = pd.read_csv('models_history.csv')
    print(df_history)
    df_history.plot()
    plt.show()

    # load the model
    my_model = load_model('house_price_model.h5')

    # evaluating the model
    from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

    predictions = my_model.predict(X_test)
    print('mean_squared_error : ')
    print(mean_squared_error(y_test, predictions))

    print('sqrt of mean_squared_error : ')
    print(np.sqrt(mean_squared_error(y_test, predictions)))

    print('mean_absolute_error : ')
    print(mean_absolute_error(y_test, predictions))

    print('explained_variance_score : ')
    print(explained_variance_score(y_test, predictions))

    plt.scatter(y_test, predictions)
    plt.show()

    # to predict a new house ( we take the first house in the data frame)
    single_house = df.drop('price', axis=1).iloc[0]
    single_house = scaler.transform(single_house.values.reshape(-1, 19))
    print(my_model.predict(single_house))
    print(df['price'][0])
