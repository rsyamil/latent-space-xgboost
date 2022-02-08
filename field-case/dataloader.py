import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size":14,"axes.titlesize":14,"axes.labelsize":14})  

import matplotlib as m
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize

#dump/read functions
def save_obj(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)
		
#function to normalize output by features: [num_data, timesteps, features]
def normalizeOutput(data):
	output_maxs = np.zeros([data.shape[1], data.shape[2]]) 
	output_maxs[:, 0] = np.max(data[:, :, 0])
	output_maxs[:, 1] = np.max(data[:, :, 1]) 
	output_maxs[:, 2] = np.max(data[:, :, 2]) 
	return (data/(output_maxs)), output_maxs
	
def normalizeProps(data):
    output_maxs = np.zeros([data.shape[1],])
    for i in range(data.shape[1]):
        output_maxs[i] = np.max(data[:, i])
    return (data/(output_maxs)), output_maxs
	
#function to plot the density plot
def histplot(data_train, data_test):
	col_names = ['$x_1$', '$x_2$', '$x_3$', '$x_4$', '$x_5$']
	plt.figure(figsize=(14, 3))
	for idx, feature in enumerate(col_names):
		plt.subplot(1, 5, idx+1)
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

		self.fileno = None
		self.field_name = None
		self.d = None
		self.u = None
		self.cumm = None
		self.loc = None
		self.x = None
		
		self.x_maxs = None
		self.d_cumm_maxs = None
		self.cumm_norm = None
		
		self.train_idx = None
		self.test_idx = None
		
	def load_data(self):
			#load cleaned data objects
			self.fileno, self.field_name, self.d, self.u, self.cumm, self.loc, self.x = load_obj('Data_Field_Bakken/DATA-clean')
			
	def calculate_cumulative_d(self, truncate=True):
		if truncate:
			self.d = self.d[:, 0:60, :]
			self.u = self.u[:, 0:60]
				
		#calculate the cumulative profiles for each well, each phase
		for i in range(self.d.shape[0]):
			for j in range(self.d.shape[2]):
				for k in range(self.d.shape[1]-1):
					self.d[i, k+1, j] = self.d[i, k+1, j] + self.d[i, k, j]
					
		#normalize by phase
		self.d, self.d_cumm_maxs = normalizeOutput(self.d)
		
		#normalize cumulative production (scalar)
		self.cumm_norm = self.cumm/np.max(self.cumm)
			
	#split data, test data has full attributes and training data may have missign values
	def get_split(self, use_complete_data_only=False):
	
		self.load_data()
		self.calculate_cumulative_d()
	
		indicator_nans = np.zeros(self.x.shape)
		indicator_nans[self.x==0] = np.nan
		indicator_nans[self.x==1] = 1
	
		nans = np.isnan(indicator_nans)
		nans_well = np.any(nans, axis=1)
		
		tot_data = self.x.shape[0]
		idx = np.linspace(0, (tot_data)-1, tot_data, dtype=np.int32)
		
		self.train_idx = idx[nans_well==True]
		self.test_idx = idx[nans_well==False]
		
		#normalize 
		self.x, self.x_maxs = normalizeProps(self.x)
		
		#prepare input data with missing values as Nans
		self.x_nans = np.copy(self.x)
		self.x_nans[self.x==0] = np.nan
		
		#prepare input data without missing values (as global mean imputation)
		self.x_imputed = np.copy(self.x)
		x_means = np.mean(self.x, axis=0)
		for i in range(self.x.shape[1]):			#by features
			for j in range(self.x.shape[0]):		#by well
				if self.x[j, i] == 0:
					self.x_imputed[j, i] = x_means[i]
					
		#only use complete dataset only, ie the test dataset (small subset, further 60:40 split)
		if use_complete_data_only:
			train_idx_complete = self.test_idx[0:200]
			test_idx_complete = self.test_idx[200:]
			
			self.train_idx = train_idx_complete
			self.test_idx = test_idx_complete
			
			x_train = self.x_imputed[train_idx_complete]
			x_test = self.x_imputed[test_idx_complete]
			
			y_train = self.d[train_idx_complete]
			y_test = self.d[test_idx_complete]
			
			cumm_train = self.cumm_norm[train_idx_complete]
			cumm_test = self.cumm_norm[test_idx_complete]
			
			return x_train, x_test, y_train, y_test, cumm_train, cumm_test
		
		x_nans_train = self.x_nans[self.train_idx]
		x_nans_test = self.x_nans[self.test_idx]
		
		x_imputed_train = self.x_imputed[self.train_idx]
		x_imputed_test = self.x_imputed[self.test_idx]
		
		y_train = self.d[self.train_idx]
		y_test = self.d[self.test_idx]
		
		cumm_train = self.cumm_norm[self.train_idx]
		cumm_test = self.cumm_norm[self.test_idx]
		
		return x_nans_train, x_nans_test, x_imputed_train, x_imputed_test, y_train, y_test, cumm_train, cumm_test
	    
    #plot fields
	def plot_fields(self):
		#get number of unique fields
		unique_fields = np.unique(self.field_name)
		no_unique_fields = len(unique_fields)
		field_code = np.arange(no_unique_fields)

		my_cmap = cm.get_cmap('Dark2')
		my_norm = Normalize(vmin=0, vmax=(no_unique_fields-1))
		cs = my_cmap(my_norm(field_code))

		#assign each data point the field code
		FieldName_idx = np.zeros(self.field_name.shape, dtype='int32')
		for i in range(len(self.field_name)):
			idx, = (np.where(unique_fields == self.field_name[i]))[0]
			FieldName_idx[i] = idx

		f = plt.figure(figsize=[10, 10])
		plt.scatter(self.loc[:,0], self.loc[:,1], s=100, c=cs[FieldName_idx])

		plt.scatter(self.loc[self.train_idx,0], self.loc[self.train_idx,1], s=50, c='k', marker='x', label='Train')
		plt.scatter(self.loc[self.test_idx,0], self.loc[self.test_idx,1], s=50, c='k', marker='|', label='Test')	

		plt.legend()
		plt.xlabel("Latitude")
		plt.ylabel("Longitude")
		plt.axis('scaled')
		plt.title(str(no_unique_fields) + " unique fields, " + str(len(self.train_idx)) + " train, "+ str(len(self.test_idx)) + " test")
		plt.grid(False)
		plt.tight_layout()
		f.savefig('readme/field_parti_maps.png', dpi=300, bbox_inches='tight')
		
	def plot_wells(self):
		plt.figure(figsize=[14, 14])
		for i in range(15):
			ax1 = plt.subplot(5, 3, i+1)

			#shift to view other sets
			i = i + 0

			_ = self.d[i]
			ax1.plot(_[:, 2], c='r', label='Gas', alpha=0.8)
			ax1.plot(_[:, 1], c='b', label='Water', alpha=0.8)
			ax1.plot(_[:, 0], c='g', label='Oil', alpha=0.8)
			ax1.legend()
			ax1.set_xlabel('Timesteps')
			ax1.set_ylabel('Rate')
			ax1.set_title(self.fileno[i])

			ax2 = ax1.twinx() 
			c = 'tab:purple'

			#plot control on another axis
			_ = self.u[i]
			ax2.plot(_, color=c)
			ax2.set_ylabel('Control', color=c) 
			ax2.tick_params(axis ='y', labelcolor=c)

			ax1.grid(False)
			ax2.grid(False)

		plt.tight_layout()

	#field data: display the output data
	def plot_data_by_phase(self):
	
		plt.figure(figsize=[3, 3])
		for i in range(self.d.shape[0]):
			plt.plot(self.d[i, :, 0], c='green', alpha=0.4)
		plt.legend(['Bakken'])
		plt.title('Oil rates')
			
		plt.figure(figsize=[3, 3])
		for i in range(self.d.shape[0]):
			plt.plot(self.d[i, :, 1], c='red', alpha=0.4)   
		plt.legend(['Bakken'])
		plt.title('Gas rates')

		plt.figure(figsize=[3, 3])
		for i in range(self.d.shape[0]):
			plt.plot(self.d[i, :, 2], c='blue', alpha=0.4)
		plt.legend(['Bakken'])
		plt.title('Water rates')
		


