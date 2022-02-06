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

class DataLoader:

	def __init__(self, verbose=False, normalized=False):
	
		self.verbose = verbose
		self.normalized = normalized

		self.fileno = None
		self.field_name = None
		self.d = None
		self.d_maxs = None
		self.u = None
		self.u_maxs = None
		self.cumm = None
		self.cumm_maxs = None
		self.loc = None
		self.x = None
		self.x_maxs = None
	
	def load_data(self):
		if self.normalized:
			#load the normalized data objects
			self.fileno, self.field_name, self.d, self.d_maxs, self.u, self.cumm, self.cumm_maxs, self.loc, self.x, self.x_maxs = load_obj('Data_Field_Bakken/DATA-norm')
		else:
			#load cleaned data objects
			self.fileno, self.field_name, self.d, self.u, self.cumm, self.loc, self.x = load_obj('Data_Field_Bakken/DATA-clean')
			
	#split data, test data has full attributes and training data may have missign values
	def get_split(self):
	
	    
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
		plt.scatter(self.loc[self.transfer_idx,0], self.loc[self.transfer_idx,1], s=50, c='k', marker='|', label='Transfer')	

		plt.legend()
		plt.xlabel("Latitude")
		plt.ylabel("Longitude")
		plt.axis('scaled')
		plt.title(str(no_unique_fields) + " unique fields, " + str(len(self.train_idx)) + " train, "+ str(len(self.transfer_idx)) + " transfer")
		plt.grid(False)
		plt.tight_layout()
		f.savefig('readme/field_parti_maps_'+self.ptype+'.png', dpi=300, bbox_inches='tight')

    
    
    