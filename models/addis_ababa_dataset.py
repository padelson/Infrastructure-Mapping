class AddisAbaba(Dataset):
	def __init__(self, filename, train_test_split = 0.8):
		data = pandas.read_csv(filename)

		self.num_binary_features = len(util.binary_features)
		self.num_continuous_features = len(util.continuous_features)

		self.y_binary = data[util.binary_features].values
		self.y_continuous = data[util.continuous_features].values

		split_value = int(len(self.y_binary) * 0.8)

		self.get_x()
		self.create_splits()

		self.size = len(self.y_binary_train)

	def create_splits(self):
		[self.y_binary_train, self.y_continuous_train, self.x_train] = map(lambda x: x[:split_value], [self.y_binary, self.y_continuous, self.x])
		[self.y_binary_test, self.y_continuous_test, self.x_test] = map(lambda x: x[split_value:], [self.y_binary, self.y_continuous, self.x])

	def num_batches(self, batch_size):
		self.num_batches = self.size / batch_size + (1 if self.size % batch_size == 0 else 0)
		self.batch_size = batch_size
		return self.num_batches

	def get_x_batch(iteration):
		if (iteration == self.num_batches-1):
			return self.x[self.batch_size * iteration :]
		else:
			return self.x[self.batch_size * iteration : self.batch_size * (iteration + 1)]

	def get_y_batch(iteration):
		if (iteration == self.num_batches-1):
			return self.y_binary[self.batch_size * iteration :], self.y_continuous[self.batch_size * iteration :]
		else:
			return self.y_binary[self.batch_size * iteration : self.batch_size * (iteration + 1)], self.y_continuous[self.batch_size * iteration : self.batch_size * (iteration + 1)]

	def get_x_test():
		return self.x_test

	def get_y_test():
		return self.y_binary_test, self.y_continuous_test