import scipy.io as sio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

dir = os.getcwd()+"/data/"
data = sio.loadmat(dir+"nn_data_rcn_kb.mat")
ws = data['ws']
phases = data['thetas']
wxs = data['wxs']
wxxs = data['wxxs']
wys = data['wys']
wyys = data['wyys']

nsamps,ny,nx = np.shape(ws)
perm = np.random.permutation(int(.9*nsamps))
training_samples = perm[:int(.8*nsamps)]
val_samples = perm[int(.8*nsamps):]
test_samples = np.arange(int(.9*nsamps), nsamps)

training_data = {'ws': ws[training_samples,:,:],
                 'wxs': wxs[training_samples,::4,::4],
                 'wxxs': wxxs[training_samples,::4,::4],
                  'wys': wys[training_samples,::4,::4],
                 'wyys': wyys[training_samples,::4,::4]}

val_data = {'ws': ws[val_samples,:,:],
                 'wxs': wxs[val_samples,::4,::4],
                 'wxxs': wxxs[val_samples,::4,::4],
                  'wys': wys[val_samples,::4,::4],
                 'wyys': wyys[val_samples,::4,::4]}

test_data = {'ws': ws[test_samples,:,:],
                 'wxs': wxs[test_samples,::4,::4],
                 'wxxs': wxxs[test_samples,::4,::4],
                  'wys': wys[test_samples,::4,::4],
                 'wyys': wyys[test_samples,::4,::4]}

w_train = training_data['ws']
w_test = test_data['ws']
w_val = val_data['ws']
w_train = w_train.astype('float32')
w_test = w_test.astype('float32')
w_val = w_val.astype('float32')
print(w_train.shape)
print(w_test.shape)
print(w_val.shape)

wx_train = training_data['wxs']
wx_train = wx_train.astype('float32')
wxx_train = training_data['wxxs']
wxx_train = wxx_train.astype('float32')
wy_train = training_data['wys']
wy_train = wy_train.astype('float32')
wyy_train = training_data['wyys']
wyy_train = wyy_train.astype('float32')

import tensorflow as tf
from tensorflow.keras import layers, losses, Model

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1024, activation='linear')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Dense(16384, activation='linear'),
            layers.Reshape((128, 128))# Adjust this based on your input dimensions
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def compute_loss(model, x, pre1, pre2, pre3, pre4):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        reconstructed = model(x)
        # First derivative
        first_derivatives = tape.gradient(reconstructed, x)
    # Second derivatives
    second_derivatives = tape.gradient(first_derivatives, x)
    del tape  # Clean up the tape

    # Mean squared error loss
    mse_loss = tf.reduce_mean(tf.square(tf.subtract(reconstructed, x)))
    # Regularization from derivatives
    thetaxx = tf.multiply(second_derivatives[:,::4,::4],tf.square(pre1)) + tf.multiply(first_derivatives[:,::4,::4],pre2)
    thetayy = tf.multiply(second_derivatives[:,::4,::4],tf.square(pre3)) + tf.multiply(first_derivatives[:,::4,::4],pre4)
    divk = thetaxx + thetayy
    thetax = tf.multiply(first_derivatives[:,::4,::4],pre1)
    thetay = tf.multiply(first_derivatives[:,::4,::4],pre3)
    ksq = tf.square(thetax)+tf.square(thetay)
    kfth = tf.square(ksq)
    self_dual_rhs = tf.square(divk)-1+2*ksq -kfth
    sd_loss = tf.reduce_mean(tf.square(self_dual_rhs))
    total_loss = mse_loss + 10 * sd_loss
    return total_loss

# Training the model
@tf.function
def train_step(model, input, pre1, pre2, pre3, pre4, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, input, pre1, pre2, pre3, pre4)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Batching the data using TensorFlow's Dataset API
batch_size = 64
main_dataset = tf.data.Dataset.from_tensor_slices(w_train).batch(batch_size)
precomputed_dataset = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(wx_train).batch(batch_size),
    tf.data.Dataset.from_tensor_slices(wxx_train).batch(batch_size),
    tf.data.Dataset.from_tensor_slices(wy_train).batch(batch_size),
    tf.data.Dataset.from_tensor_slices(wyy_train).batch(batch_size),
))

# Combine main data with precomputed data
combined_dataset = tf.data.Dataset.zip((main_dataset, precomputed_dataset))

# Define the model, optimizer, and train function if not already defined
autoencoder = Autoencoder()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Function to train the model
def train(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            input_data, precomputed_data = batch
            loss = train_step(autoencoder, input_data, *precomputed_data, optimizer)
            print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Start training
train(combined_dataset, epochs=100)

print("debug")
encoded_images = autoencoder.encoder(w_test).numpy()
print("debug")