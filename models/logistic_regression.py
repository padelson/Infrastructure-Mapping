from keras.models import Sequential 
from keras.layers import Dense, Activation 

def get_logreg_model(input_dim):
	output_dim = 2 # Binary
	model = Sequential() 
	model.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
	return model

# Returns [Training Score, Training Accuracy], [Test score, Test Accuracy]
def train_model(model, X, Y, batch_size, num_epoch, validation_split = 0.8):
	X_train, X_test = X[:0.8*len(X)], X[0.8*len(X):]
	Y_train, Y_test = Y[:0.8(len(Y)), Y[0.8*len(Y):]

	history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epoch,verbose=1, validation_data=(X_test, Y_test))

	score_train = model.evaluate(X_train, Y_train, verbose=0) 
	score_test = model.evaluate(X_test, Y_test, verbose=0) 

	return score_train, score_test