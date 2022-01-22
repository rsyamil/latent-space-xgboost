import util

import numpy as np
import keras
from keras.models import Sequential, Model

from keras.layers import Layer, Lambda, Reshape, LeakyReLU
from keras.layers import Dense, Dropout, Flatten, Multiply
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras import backend as K

from keras.engine.base_layer import InputSpec
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras import regularizers, activations, initializers, constraints
from keras.constraints import Constraint

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History 

from IPython.display import clear_output

from keras.utils import plot_model
from keras.models import load_model

class Models:

	def __init__(self, name=[], x_dim=8, z_dim=3, timesteps=12, n_features=3, verbose=False):
		self.name = name
		self.d2d = []
		self.d2zd = []
		self.zd2d = []
		self.z_dim = z_dim
		self.x_dim = x_dim
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
    
		_ = Dense(24*2)(encoded_d)
		_ = Reshape((3, 8*2))(_)

		_ = Conv1D(8*2, 6, padding="same")(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		_ = Conv1D(16*2, 3)(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		_ = Conv1D(16*2, 2)(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		decoded_d = Conv1D(3, 3, padding='valid')(_)

		return decoded_d

	def train_autoencoder1D(self, x_train, d_train, load = False, epoch=200):

		#data encoder
		input_dt, encoded_d = self.encoder1D()
		decoded_d = self.decoder1D(encoded_d)

		self.d2d = Model(input_dt, decoded_d)
		opt = keras.optimizers.Adam(lr=1e-3)
		self.d2d.compile(optimizer=opt, loss="mse", metrics=['mse'])
		self.d2d.summary()

		#train the neural network alternatingly
		totalEpoch = epoch
		plot_losses1 = Util.PlotLosses()
		history1 = History()
		d2d_loss = np.zeros([totalEpoch, 4])
    
		for i in range(totalEpoch):

			#train data recons AE
			self.d2d.fit(d_train, d_train,        
					epochs=1,
					batch_size=128,
					shuffle=True,
					validation_split=0.2,
					callbacks=[plot_losses1, EarlyStopping(monitor='loss', patience=60), history1])

			#copy loss
			d2d_loss[i, :] = np.squeeze(np.asarray(list(history1.history.values())))

			#write to folder for every 10th epoch for monitoring
			figs = util.plotAllLosses(d2d_loss)
			figs.savefig('readme/AE_Losses.png')
			
		#set the encoder model
		self.d2zd = Model(input_dt, encoded_d)

		#set the decoder model
		zd_dec = Input(shape=(self.z_dim, )) 
		_ = self.x2d.layers[10](zd_dec)
		for i in range(11, 21):
			_ = self.x2d.layers[i](_)
		decoded_dt_ = self.x2d.layers[21](_)
		self.zd2d = Model(zd_dec, decoded_dt_)

    
    
    
    
    
    
    
    
 