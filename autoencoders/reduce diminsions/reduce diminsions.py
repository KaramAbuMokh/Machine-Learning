import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# new : to make data sets
from sklearn.datasets import make_blobs

if __name__ == '__main__':

    # create the data set
    data=make_blobs(n_samples=300
                    ,n_features=2
                    ,centers=2  # how many clusters
                    ,cluster_std=2  # how much noise
                    ,random_state=101)

    # the data now is contains 2 features
    # and every 2 features they are clustered
    # to 2 clusters, 0 or 1

    # x the features and y the label
    X,y=data

    # create noise
    np.random.seed(seed=101)
    z_noise=np.random.normal(size=len(X))
    z_noise=pd.Series(z_noise)

    # put the features in data frame
    feat=pd.DataFrame(X)

    # concat the features with the noise
    feat=pd.concat([feat,z_noise],axis=1)
    feat.columns=['X1','X2','X3']

    # show the x1,x2 in points in 2 dims
    plt.scatter(feat['X1'],feat['X2'],c=y)
    plt.show()

    from mpl_toolkits.mplot3d import axes3d
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(feat['X1'],feat['X2'],feat['X3'],c=y)
    plt.show()

    # scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(feat)

    
    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # for playing with the learning rate
    from tensorflow.keras.optimizers import SGD

    # 3 --> 2 --> 3

    encoder=Sequential()
    encoder.add(Dense(units=2
                      ,activation='relu'
                      ,input_shape=[3]))

    decoder=Sequential()
    decoder.add(Dense(units=3
                      ,activation='relu'
                      ,input_shape=[2]))

    autoencoder=Sequential([encoder,decoder])

    autoencoder.compile(loss='mse'
                        ,optimizer=SGD(lr=1.5))

    # train the model
    autoencoder.fit(scaled_data,scaled_data,epochs=5)

    # save the model
    autoencoder.save('model to reduce dims.h5')
    # save the encoder
    encoder.save('encoder to reduce dims.h5')
    # save the model
    decoder.save('decoder to reduce dims.h5')


    # load the model
    from tensorflow.keras.models import load_model
    my_model=load_model('model to reduce dims.h5')
    encoder = load_model('encoder to reduce dims.h5')
    decoder = load_model('decoder to reduce dims.h5')


    # the encoded data
    encoded_2dim=encoder.predict(scaled_data)
    print(encoded_2dim)
    plt.scatter(encoded_2dim[:,0],encoded_2dim[:,1])
    plt.show()














