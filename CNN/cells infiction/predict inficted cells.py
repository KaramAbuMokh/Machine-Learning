import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = '../../DATA//cell_images'

# show content of directory


def dir_content(data_dir):
    print(os.listdir(data_dir))
# dir_content(data_dir)


if __name__ == '__main__':

    # to read images from the pc
    from matplotlib.image import imread

    # set the pathes for the test and train directories
    test_path = data_dir+'/test'
    train_path = data_dir + '/train'

    # read image for infected cell
    para_cell = train_path+'/parasitized'
    para_cell = para_cell+'/'+os.listdir(para_cell)[0]
    # print(para_cell)
    # imread() is for read the image matrix
    # and this function take the path of the image
    # print(imread(para_cell))
    # imshow() function to display the image
    # and its take the image matrix
    # that means the values of the pixels
    # plt.imshow(imread(para_cell))
    # plt.show()

    # read image for uninfected cell
    uninfected_cell = train_path+'/uninfected'
    uninfected_cell = uninfected_cell+'/'+os.listdir(uninfected_cell)[0]
    # imread() is for read the image matrix
    # and this function take the path of the image
    uninfected_cell = imread(uninfected_cell)
    # imshow() function to display the image
    # and its take the image matrix
    # that means the values of the pixels
    # plt.imshow(uninfected_cell)
    # plt.show()

    # read the images into a dataframe or matrix
    dim1 = []
    dim2 = []

    for image_name in os.listdir(test_path+'/uninfected'):
        img = imread(test_path+'/uninfected/'+image_name)
        d1, d2, colors = img.shape
        dim1.append(d1)
        dim2.append(d2)
    # if we print d1 and d2 we can see that the images not the same size

    # resize the images and image processing

    # the mean of the size of the images is (130,130)
    print(np.mean(dim1))
    print(np.mean(dim2))

    image_shape = (130, 130, 3)

    # image processing
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # rotation_range :              the angle of the rotation
    # width_shift_range=0.1 :       shift the width of the image in the
    #                               range of 0 to 0.1 of the image width
    # height_shift_range=0.1 :      shift the height of the image in the
    #                               range of 0 to 0.1 of the image height
    # rescale=1/255 : to make the values between 0 and 1 but they
    #                               already are between 0 and 1 so
    #                               now we will not put this parameter
    # shear_range=0.1 :             cutting away part of the image
    # fill_mode='neares'  :         when doing transformation
    #                               fill the space with nearest points
    image_gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')

    # show inficted image before the transform
    para_img = imread(para_cell)
    plt.imshow(para_img)
    plt.show()
    print(para_img)
    # transform the image
    print(para_img.shape)
    transformed_img = image_gen.random_transform(para_img)

    # show inficted image before the transform
    plt.imshow(transformed_img)
    plt.show()

    # generate the images and load to RAM
    # image_gen.flow_from_directory((train_path))
    # image_gen.flow_from_directory((test_path))

    batch_size = 16

    # split the set of the images
    train_img_gen = image_gen.flow_from_directory(train_path,
                                                  target_size=image_shape[:2], color_mode='rgb', batch_size=batch_size, class_mode='binary')
    test_img_gen = image_gen.flow_from_directory(test_path,
                                                 target_size=image_shape[:2], color_mode='rgb', batch_size=batch_size, class_mode='binary', shuffle=False)
    # printing the results classes

    # creating the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              input_shape=image_shape, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # add early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='lval-oss', patience=2)

    print(f'The results classes : {train_img_gen.class_indices}')

    # train the model
    model.fit(train_img_gen, epochs=20,
              validation_data=test_img_gen, callbacks=[early_stop])

    # saving the model
    model.save('model to predict infected cells.h5')

    # save the history
    history_of_the_model = pd.DataFrame(model.history.history)
    history_of_the_model.to_csv('the history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('model to predict infected cells.h5')

    # load the history of the model
    history_of_the_model = pd.read_csv('the history of the model.csv')

    history_of_the_model.plot()
    plt.show()

    # predict the test set
    pred = my_model.predict_generator(test_img_gen)
    predictions = pred > 0.8

    # evaluation
    from sklearn.metrics import classification_report, confusion_matrix
    # the report
    print(classification_report(test_img_gen.classes, predictions))

    # the report
    print(confusion_matrix(test_img_gen.classes, predictions))

    # print the classes
    print(train_img_gen.class_indices)

    # load image
    from tensorflow.keras.preprocessing import image
    print(para_cell)  # printing the image path
    my_img = image.load_img(para_cell, target_size=image_shape)

    # transform the image to array
    my_img_arr = image.img_to_array(my_img)

    # show the image
    para_img = imread(para_cell)
    plt.imshow(para_img)
    plt.show()

    # print the image shape
    print(
        f'the image shape after loading with shape (130,130,3) is :{my_img_arr.shape}')

    # reshape the array to (1,130,130,3)
    my_img_arr = np.expand_dims(my_img_arr, axis=0)

    # print the image shape after expanding
    print(f'the image shape after expanding is :{my_img_arr.shape}')

    # predict the image
    print(my_model.predict(my_img_arr))
