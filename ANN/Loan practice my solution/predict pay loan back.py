import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# read the data
def read_data():
    df = pd.read_csv('../../DATA/lending_club_loan_two.csv')
    print(df.head().to_string())
    return df


# read_data()

# --------------------------------------------------------

# count plot
def counts():
    plt.figure(figsize=(12, 6))
    df = read_data()
    sub = sorted(df['sub_grade'].unique())
    sns.countplot('sub_grade', data=df, order=sub)
    plt.show()


# counts()

# --------------------------------------------------------

# some info about the data
def informations(df):
    print(df.isnull().sum())
    print(df.info())


# informations(read_data())


# --------------------------------------------------------

# convert results to 0,1
def convert_results(str):
    if str == 'Fully Paid':
        return 1
    else:
        return 0


# --------------------------------------------------------

# loan amount histogram
def amount_histogram(df):
    sns.distplot(df['loan_amnt'], kde=False, bins=40)
    # bins means how many values to display
    plt.show()


# amount_histogram(read_data())

# --------------------------------------------------------

# correlations
def corrs(df):
    print(df.corr().to_string())
    plt.figure(figsize=(12, 7))
    sns.heatmap(df.corr(), annot=True, cmap='viridis')
    plt.show()


# corrs(read_data())

# --------------------------------------------------------


# informations about the amount groupby the status
def groupby_status_vs_amount(df):
    print(df.groupby('loan_status')['loan_amnt'].describe().to_string())


# groupby_status_vs_amount(read_data())

# --------------------------------------------------------

# cont grade compare to the status
def grade_vs_status(df):
    sns.countplot('grade', hue='loan_status', data=df)
    plt.show()


# grade_vs_status(read_data())

# --------------------------------------------------------

# the percentage of the missing data
def missing_data_percent():
    df = read_data()
    print(100 * (df.isnull().sum() / len(df)))


# missing_data_percent()


# --------------------------------------------------------

# sort the employment length
def sort_emp_length():
    df = read_data()
    print(sorted(df['emp_length'].dropna().unique()))


# sort_emp_length()

# plot the sorted employment length vs the status
def plot_sort_emp_length():
    df = read_data()
    plt.figure(figsize=(12, 4))
    emp_len_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years',
                     '9 years', '10+ years']
    sns.countplot(x='emp_length', data=df,
                  order=emp_len_order, hue='loan_status')
    plt.show()


# plot_sort_emp_length()

# --------------------------------------------------------

# the percentage of people who
# didnt pay back the loan vs
# the employment length
def percent_charged_vs_emp_length():
    df = read_data()
    count_charged_off_per_emp_length = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()[
        'loan_status']
    count_fully_paid_per_emp_length = df[df['loan_status'] == 'Fully Paid'].groupby(
        'emp_length').count()['loan_status']
    perc = count_charged_off_per_emp_length / \
        (count_charged_off_per_emp_length + count_fully_paid_per_emp_length)
    print(perc)
    perc.plot(kind='bar')
    plt.show()


# percent_charged_vs_emp_length()

# --------------------------------------------------------

# the correlation between the mort_acc with the others
def mort_acc_correlations():
    df = read_data()
    print(df.corr()['mort_acc'].sort_values())


# mort_acc_correlations()


# ********************************************************


# function to fill the mort_acc
def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# --------------------------------------------------------

# change the term to integer
def convert_term(df):
    df['term'] = df['term'].apply(lambda str: int(str[:3]))
    print(df.head().to_string())
    return df


# convert_term(read_data())

if __name__ == '__main__':

    # red the data
    df = read_data()

    # convert the results to 0 or 1
    df['loan_status'] = df['loan_status'].apply(convert_results)
    # print(df.head().to_string())

    # dropping the employment title
    df = df.drop('emp_title', axis=1)

    # dropping the employment length
    df = df.drop('emp_length', axis=1)

    # drop the title because its sub value of the purpose
    df = df.drop('title', axis=1)

    # we can see that the total_acc correlated with the mort_acc
    # so we will group by total accounts and find the mean of the mort_acc per total account
    total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
    df['mort_acc'] = df.apply(lambda x: fill_mort_acc(
        x['total_acc'], x['mort_acc']), axis=1)

    # now there is small percent of data that is null
    # we can just drop the rows
    df = df.dropna()

    # convert the term to integer
    df = convert_term(df)

    # drop the grade column because we have sub_grade
    df = df.drop('grade', axis=1)

    # get the dummies and concat with the data frame
    # dummies is actually to build matrix with the unique values
    # 1 if the row have this unique value and 0 if not
    # dropping the first because if no unique value have 1 the the first value was 1
    dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
    # print(dummies)
    df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

    # do the same for the others
    dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                             drop_first=True)
    # print(dummies)
    df = pd.concat(
        [df.drop(['verification_status', 'application_type',
                 'initial_list_status', 'purpose'], axis=1), dummies],
        axis=1)

    # in the home_ownership we want to put the rows that have
    # the values NONE and ANY in the OTHER category because they
    # are small amount of people
    df['home_ownership'] = df['home_ownership'].replace(
        ['NONE', 'ANY'], 'OTHER')

    # do the dummy thing with the home_ownership label
    dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
    df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

    # extract the zip code from the address
    df['zip_code'] = df['address'].apply(lambda x: x[-5:])

    # do the dummy thing with the zip_code label
    dummies = pd.get_dummies(df['zip_code'], drop_first=True)
    df = pd.concat([df.drop('zip_code', axis=1), dummies], axis=1)

    # drop the address column
    df = df.drop('address', axis=1)

    # drop the issue_d column witch is the date of taking the loan
    df = df.drop('issue_d', axis=1)

    # convert the earliest_cr_line to int
    df['earliest_cr_line'] = df['earliest_cr_line'].apply(
        lambda date: int(date[-4:]))

    # grab a sample from the data
    # df=df.sample(frac=0.2,random_state=101)

    # split the result
    X = df.drop('loan_status', axis=1).values
    y = df['loan_status'].values

    # split the test and train sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101)

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # creating the model
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential()
    model.add(Dense(78, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(39, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(19, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x=X_train, y=y_train, epochs=25, batch_size=256,
              validation_data=(X_test, y_test))

    # save the model and the history
    model.save('model to predict paying loan back.h5')
    history_of_model = pd.DataFrame(model.history.history)
    history_of_model.to_csv(
        'history of paying loan back model.csv', index=False)

    # load the model and the history
    history_of_model = pd.read_csv('history of paying loan back model.csv')
    from tensorflow.keras.models import load_model
    my_model = load_model('model to predict paying loan back.h5')

    # plot the losses
    history_of_model.plot()
    plt.show()

    # evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    predictions = my_model.predict_classes(X_test)
    print(classification_report(y_test, predictions))
    # precision : percent of the true 1 prediction from all of the 1 predictions
    # recall : percent of the true 1 predictions from all the actual 1 values
