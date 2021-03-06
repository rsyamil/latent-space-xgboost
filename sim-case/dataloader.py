import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as m

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size":14,"axes.titlesize":14,"axes.labelsize":14})  

#function to plot the density plot
def histplot(data_train, data_test):
	col_names = ['Perm. (mD)', 'Poro. (ratio)', 'SW (ratio)', 'Thick. (m)', 'Press. (psi)', 'Density (kgm3)']
	plt.figure(figsize=(9, 6))
	for idx, feature in enumerate(col_names):
		plt.subplot(2, 3, idx+1)
		dtr = data_train[:, idx]
		dts = data_test[:, idx]
		dtr = dtr[~np.isnan(dtr)]	#drop existing nans
		dts = dts[~np.isnan(dts)]
		new_bins = np.linspace(np.min(dtr), np.max(dtr), 30)
		sns.distplot(dtr, hist=True, bins=new_bins, norm_hist=False, kde=False, label="Train")
		sns.distplot(dts, hist=True, bins=new_bins, norm_hist=False, kde=False, label="Test")
		plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='off')
		plt.tight_layout(), plt.legend(), plt.title(feature)

class DataLoader:

	def __init__(self, verbose=False):

		self.verbose = verbose

		self.x = []         	#(600, 6)
		self.y = []         	#(600, 60, 3)
		self.y_cumm = []		#(600, 1)
		
		self.x_raw = []         #(600, 6)
		self.y_raw = []         #(600, 60, 3)
		self.y_cumm_raw = []	#(600, 1)

		self.x_min = 0
		self.x_max = 0
		self.y_min = np.array ([])   #(3,) for each channel
		self.y_max = np.array ([])   #(3,)
		
		self.x_means = []
        
	def normalize_x(self):
		self.x_min = np.min(self.x, axis=0)
		self.x_max = np.max(self.x, axis=0)
		self.x = (self.x - self.x_min)/(self.x_max - self.x_min)
    
	def normalize_y(self):
		'''normalize by channel'''
		n_features = self.y.shape[-1]
		for f in range(n_features):
			self.y_min = np.append(self.y_min, np.min(self.y[:,:,f]))
			self.y_max = np.append(self.y_max, np.max(self.y[:,:,f]))
			self.y[:,:,f] = (self.y[:,:,f] - self.y_min[f])/(self.y_max[f] - self.y_min[f])

	def load_data(self):
    
		df = pd.read_csv("Data_Simulated_Bakken/DATA_BAKKEN.csv")
		df = df.to_numpy()

		#original data
		self.x = df[:, 0:6]
		oil = df[:, 6:66]
		water = df[:, 68:128]
		gas = df[:, 128:188]
		self.y = np.stack((oil, water, gas), axis=2)
		
		#calculate cumulative oil
		self.y_cumm_raw = np.sum(oil, axis=1)
		self.y_cumm = (self.y_cumm_raw - np.min(self.y_cumm_raw))/(np.max(self.y_cumm_raw) - np.min(self.y_cumm_raw))

		#shuffle x and y together, since theyre from the same provenance! 
		np.random.seed(77)
		shuffle_idx = np.random.permutation(self.x.shape[0])

		#shuffle data
		self.x = self.x[shuffle_idx]
		self.y = self.y[shuffle_idx]

		#make copies (for spatial plotting) and normalize
		self.x_raw = np.copy(self.x)
		self.normalize_x()

		self.y_raw = np.copy(self.y)
		self.normalize_y()
		
		#calculate the mean
		self.x_means = np.mean(self.x, axis=0)

	def get_data_split(self, split=0.8):
	
		self.load_data()

		tot_data = self.x.shape[0]
		idx = np.linspace(0, (tot_data)-1, tot_data, dtype=np.int32)
		partition = int(tot_data*split)
		self.train_idx = idx[0:partition]
		self.test_idx = idx[partition:]

		x_train = self.x[self.train_idx]
		x_test = self.x[self.test_idx]
		y_train = self.y[self.train_idx]
		y_test = self.y[self.test_idx]
		
		y_cumm_train = self.y_cumm[self.train_idx]
		y_cumm_test = self.y_cumm[self.test_idx]

		return x_train, x_test, y_train, y_test, y_cumm_train, y_cumm_test
    
        

    
    
    