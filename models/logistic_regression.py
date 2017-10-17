from keras.models import Sequential 
from keras.layers import Dense, Activation 
from sklearn.metrics import f1_score

def f1_score_metric(y_true, y_pred, average='micro'):
	return f1_score(y_true, y_pred)

def logreg_model(input_dim):
	output_dim = 2 # Binary
	model = Sequential() 
	model.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
	model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
	return model

# Expects: logreg_model, [num_examples, length of band], [num_examples, 1], batch size, number epochs
# Returns [Training Score, Training Accuracy], [Test score, Test Accuracy]
def train_model(model, X, Y, batch_size, num_epoch):
	split = int(0.8*len(X))
	X_train, X_test = X[:split], X[split:]
	Y_train, Y_test = Y[:split], Y[split:]

	history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epoch,verbose=1, validation_data=(X_test, Y_test))

	score_train = model.evaluate(X_train, Y_train, verbose=0) 
	score_test = model.evaluate(X_test, Y_test, verbose=0) 

	return score_train, score_test