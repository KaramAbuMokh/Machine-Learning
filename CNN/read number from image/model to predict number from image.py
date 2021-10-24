import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

# load the data
def load_data():
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    return (X_train,y_train),(X_test,y_test)

# --------------------------------------------------


# show the shape of column
def shape_of(df):
    print(df.shape)

# --------------------------------------------------


# show the image
def show_image(single_image):
    plt.imshow(single_image)
    plt.show()

if __name__ == '__main__':

    # load the data
    (X_train, y_train), (X_test, y_test)=load_data()

    # shape_of(X_train) --> that will print : (60000, 28, 28)

    # grab single image and show it
    # show_image(X_train[0])

    # convert the y_train from numbers to binary
    from tensorflow.keras.utils import to_categorical
    y_cat_train=to_categorical(y_train,num_classes=10)
    y_cat_test = to_categorical(y_test,num_classes=10)
    # that will convert 3 --to-->[0,0,0,1,0,0,0,0,0,0]

    # scale the values
    X_train=X_train/255
    X_test=X_test/255

    # add one dimension to notice the model
    # that the image is grayscale image
    # witch is the dimension of the color channel
    X_train=X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten


    
    
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(4,4),
                     input_shape=(28,28,1),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam'
                  ,metrics=['accuracy'])

    # for the early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop=EarlyStopping(monitor='val_loss',patience=1)
    model.fit(X_train,y_cat_train,epochs=10,validation_data=(X_test,y_cat_test)
              ,callbacks=early_stop)

    # save the model and the history
    model.save('model of recognizing the numbers.h5')
    models_history=pd.DataFrame(model.history.history)
    models_history.to_csv('history of the model.csv',index=False)
    
    

    # loading the model
    from tensorflow.keras.models import load_model
    models_history=pd.read_csv('history of the model.csv')
    my_model=load_model('model of recognizing the numbers.h5')

    models_history[['loss','val_loss']].plot()
    plt.show()
    models_history[['accuracy', 'val_accuracy']].plot()
    plt.show()

    # see the metrics names
    print(my_model.metrics_names)

    # see the metrics values
    print(my_model.evaluate(X_test,y_cat_test,verbose=0))

    # predict the test set
    predictions = my_model.predict_classes(X_test)

    # report
    from sklearn.metrics import classification_report,confusion_matrix

    print(classification_report(y_test,predictions))

    # confusion_matrix
    print(confusion_matrix(y_test,predictions))

    # predict image from the test set
    my_number=X_test[0]
    plt.imshow(my_number.reshape(28,28))
    plt.show()
    print(my_model.predict_classes(my_number.reshape(1,28,28,1)))



