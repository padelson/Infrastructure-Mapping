import abc

class Dataset(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def num_batches(self, batch_size):
		""" 
		Returns the number of training batches
		"""
	@abc.abstractmethod
	def get_x_batch(self, i):
		"""
		Returns input batch i
		"""

	@abc.abstractmethod
	def get_y_batch(self, i):
		"""
		Returns output batch i as a tuple (category ndarray, continuous ndarray)
		"""

	@abc.abstractmethod
	def get_x_test(self):
		"""
		Returns the entire test input ndarray
		"""

	@abc.abstractmethod
	def get_y_test(self):
		"""
		Returns the entire test output ndarray as a tuple (category ndarray, continuous ndarray)
		"""

	@abc.abstractmethod
	def num_binary_features(self):
		"""
		Returns the number of binary features
		"""

	@abc.abstractmethod
	def num_continuous_features(self):
		"""
		Returns the number of continuous features
		"""


