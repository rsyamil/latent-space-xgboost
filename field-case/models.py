import util

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Reshape, LeakyReLU
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers, activations, initializers, constraints
from tensorflow.keras.constraints import Constraint

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History 

from IPython.display import clear_output

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class models:

	def __init__(self, name=[], z_dim=3, timesteps=12, n_features=3, verbose=False):
		self.name = name
		self.d2d = []
		self.d2zd = []
		self.zd2d = []
		self.z_dim = z_dim
		self.timesteps = timesteps
		self.n_features = n_features
        
	def encoder1D(self):
		#define the simple autoencoder
		input_dt = Input(shape=(self.timesteps, self.n_features))  

		# define data encoder
		_ = Conv1D(16*2, 3, padding='same')(input_dt)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = MaxPooling1D(2, padding="same")(_)

		_ = Conv1D(8*2, 6, padding='same')(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = MaxPooling1D(2, padding="same")(_)

		_ = Flatten()(_)
		encoded_d = Dense(self.z_dim)(_)

		return input_dt, encoded_d
        
	def decoder1D(self, encoded_d):
    
		_ = Dense(240)(encoded_d)
		_ = Reshape((15, 16))(_)

		_ = Conv1D(8*2, 6, padding="same")(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		_ = Conv1D(16*2, 3, padding="same")(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		decoded_d = Conv1D(3, 3, padding='same')(_)

		return decoded_d

	def train_autoencoder1D(self, d_train, load = False, epoch=200):

		#data encoder
		input_dt, encoded_d = self.encoder1D()
		decoded_d = self.decoder1D(encoded_d)

		self.d2d = Model(input_dt, decoded_d)
		opt = keras.optimizers.Adam(lr=1e-3)
		self.d2d.compile(optimizer=opt, loss="mse", metrics=['mse'])
		self.d2d.summary()

		#train the neural network alternatingly
		plot_losses1 = util.PlotLosses()
		history1 = History()

		#train data recons AE
		self.d2d.fit(d_train, d_train,        
				epochs = epoch,
				batch_size = 128,
				shuffle = True,
				validation_split = 0.2,
				callbacks = [plot_losses1, EarlyStopping(monitor='loss', patience=60), history1])

		#set the encoder model
		self.d2zd = Model(input_dt, encoded_d)

		#set the decoder model
		zd_dec = Input(shape=(self.z_dim, )) 
		_ = self.d2d.layers[9](zd_dec)
		for i in range(10, 17):
			_ = self.d2d.layers[i](_)
		decoded_dt_ = self.d2d.layers[17](_)
		self.zd2d = Model(zd_dec, decoded_dt_)

    
    
    
    
    
    
    
    
 