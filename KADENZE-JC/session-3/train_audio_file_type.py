

def train_audio_file_type(img,
          learning_rate=0.001,
          batch_size=200,
          n_iterations=10,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):

    ############## AUDIO FILE TRANSFORMATION PROCESS ##################
    # Store every magnitude frame and its label of being music: 0 or speech: 1
	Xs, ys = [], []

	tf.reset_default_graph()
	
	# Create the input to the network.  This is a 4-dimensional tensor!
	X = tf.placeholder(name='X', shape=[None, 43, 256, 1], dtype=tf.float32)
	
	# Create the output to the network.  This is our one hot encoding of 2 possible values (TODO)!
	Y = tf.placeholder(name='Y', shape=[None, 2], dtype=tf.float32)
	n_filters = [9, 9, 9, 9]
	
	# Now let's loop over our n_filters and create the deep convolutional neural network
	H = X
	for layer_i, n_filters_i in enumerate(n_filters):
		
		# Let's use the helper function to create our connection to the next layer:
		# TODO: explore changing the parameters here:
		H, W = utils.conv2d(
			H, n_filters_i, k_h=3, k_w=3, d_h=2, d_w=2,
			name=str(layer_i))
		
		# And use a nonlinearity
		# TODO: explore changing the activation here:
		H = tf.nn.relu(H)
		#H = tf.nn.tanh(H)
		#H = tf.nn.sigmoid(H)
		#H = tf.nn.softmax(H)
		
		# Just to check what's happening:
		print(H.get_shape().as_list())
	
	# Connect the last convolutional layer to a fully connected network (TODO)!
	fc, W = utils.linear(H, 100, activation=tf.nn.relu, name='fc_1')

	# And another fully connected layer, now with just 2 outputs, the number of outputs that our
	Y_pred, W = utils.linear(fc, 2, activation=tf.nn.softmax, name='fc_2')
	
	loss = utils.binary_cross_entropy(Y_pred, Y)
	cost = tf.reduce_mean(tf.reduce_sum(loss, 1))
	
	predicted_y = tf.argmax(Y_pred, 1)
	actual_y = tf.argmax(Y, 1)
	correct_prediction = tf.equal(predicted_y, actual_y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
	
	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	
	
	n_epochs = 25
	batch_size = 50

	# Create a session and init!
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# Now iterate over our dataset n_epoch times
	for epoch_i in range(n_epochs):
		print('Epoch: ', epoch_i)
		
		# Train
		this_accuracy = 0
		its = 0
		
		# Do our mini batches:
		for Xs_i, ys_i in ds.train.next_batch(batch_size):
			# Note here: we are running the optimizer so
			# that the network parameters train!
			this_accuracy += sess.run([accuracy, optimizer], feed_dict={
					X:Xs_i, Y:ys_i})[0]
			its += 1
			print(this_accuracy / its)
		print('Training accuracy: ', this_accuracy / its)
		
		# Validation (see how the network does on unseen data).
		this_accuracy = 0
		its = 0
		
		# Do our mini batches:
		for Xs_i, ys_i in ds.valid.next_batch(batch_size):
			# Note here: we are NOT running the optimizer!
			# we only measure the accuracy!
			this_accuracy += sess.run(accuracy, feed_dict={
					X:Xs_i, Y:ys_i})
			its += 1
		print('Validation accuracy: ', this_accuracy / its)
		
	# Print final test accuracy:
	test = ds.test
	print("Final test accuracy : ", sess.run(accuracy,
				   feed_dict={
					   X: test.images,
					   Y: test.labels
				   }))
	
	
	
	
	
	
	
	
	
	