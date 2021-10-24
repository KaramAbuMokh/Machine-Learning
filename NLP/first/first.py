import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':

    # read the test
    text = open('../../Data/shakespeare.txt', 'r').read()

    # the unique characters
    vocab = sorted(set(text))

    # ---------------

    # text processing

    # give number to every character in a dictionary
    char_to_ind = {char: ind for ind, char in enumerate(vocab)}

    # put the characters in array
    ind_to_char = np.array(vocab)

    # translate the text to numbers and put every number (char) in cell in array
    encoded_text = np.array([char_to_ind[c] for c in text])

    # creating the batches
    seq_len = 120
    total_num_seq = len(text)//(seq_len+1)
    char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
    sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

    # creating the targets
    def create_seq_targets(seq):
        input_txt = seq[:-1]
        target_txt = seq[1:]
        return input_txt, target_txt

    # final data set
    dataset = sequences.map(create_seq_targets)

    '''
    
    # print a sample of the dataset
    for input_txt, target_txt in dataset.take(1):
        print(input_txt.numpy())
        print(''.join(ind_to_char[input_txt.numpy()]))
        print('\n')
        print(target_txt.numpy())
        print(''.join(ind_to_char[target_txt.numpy()]))

    '''

    batch_size = 128  # feeding 128 sets in a time

    # because the set is large
    buffer_size = 10000

    # shuffle the dataset
    dataset = dataset.shuffle(buffer_size == buffer_size).batch(
        batch_size, drop_remainder=True)

    # set some variables
    vocab_size = len(vocab)       # the unique characters = 84
    embed_dim = 64                # close to the vocab size, thats for the embeding layer
    rnn_neurons = 1026

    # that will creat loss function for us
    from tensorflow.keras.losses import sparse_categorical_crossentropy

    def sparse_cat_loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    # create the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, GRU, Dense

    def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):

        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim,
                  batch_input_shape=[batch_size, None]))
        model.add(GRU(rnn_neurons, return_sequences=True,
                  stateful=True, recurrent_initializer='glorot_uniform'))
        model.add(Dense(vocab_size))
        model.compile(optimizer='adam', loss=sparse_cat_loss)

        return model

    model = create_model(vocab_size=vocab_size, embed_dim=embed_dim,
                         rnn_neurons=rnn_neurons, batch_size=batch_size)

    # train the model
    model.fit(dataset, epochs=30)

    # save the model
    model.save('lang model.h5')

    # save the history
    his = pd.DataFrame(model.history.history)
    his.to_csv('history of the model.csv', index=False)

    # load the model
    from tensorflow.keras.models import load_model
    my_model = load_model('lang model.h5')

    # load the history
    models_history = pd.read_csv('history of the model.csv')

    models_history.plot()
    plt.show()

    # to predict

    for input_example_batch, target_example_batch in dataset.take(1):
        exampl_batch_pred = model(input_example_batch)

    sample_indices = tf.random.categorical(exampl_batch_pred[0], num_samples=1)

    sample_indices = tf.squeeze(sample_indices, axis=-1).numpy()

    print(ind_to_char[sample_indices])

    # to predict
    from tensorflow.keras.models import load_model

    model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
    model.load_weights('shakespeare_gen.h5')
    model.build(tf.TensorShape([1, None]))

    def generate_text(model, start_seed, gen_size=500, temp=1.0):
        num_generate = gen_size
        input_eval = [char_to_ind[s] for s in start_seed]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        temperature = temp
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(ind_to_char[predicted_id])
        return (start_seed + ''.join(text_generated))

    print(generate_text(model, 'JULIET', gen_size=1000))
