import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt

#function to view training and validation losses
class PlotLosses(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		self.fig = plt.figure()
		self.logs = []

	def on_epoch_end(self, epoch, logs={}):
		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.i += 1
		
		print(len(self.losses))
		
		clear_output(wait=True)
		plt.plot(self.x, self.losses, label="loss", c = 'green')
		plt.plot(self.x, self.val_losses, label="val_loss", c = 'red')
		plt.legend()
		plt.show()
        
#function to view multiple losses
def plotAllLosses(loss1):         
	N, m1f = loss1.shape

	print(loss1.shape)

	fig = plt.figure(figsize=(6, 6))
	plt.subplot(2, 1, 1)
	plt.plot(loss1[:, 0], label='l1', linewidth=3)
	plt.plot(loss1[:, 1], label='l2')
	plt.legend()

	return fig