# I created here a function to directly paint from one image, based on the part two.
# The advantage is that with this function I plot the paint progress as in part two, and this helps to fastly tune the 
# parameters and have a better intuition about what parameters could be good with the image.

def train_paint_with_one_image(img,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=10,
          gif_step=2,
          n_neurons=30,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):

    ############## IMAGE TRANSFORMATION PROCESS ##################
    xs, ys = split_image(img)
    xs = np.divide(xs - np.mean(xs), np.std(xs))
    ys = ys / 255.0
    
    """
    # Let's build the model using the build_model function
    model2 = build_model(xs, ys, n_neurons, n_layers, activation_fn, final_activation_fn, cost_type)
    # return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}
    X = model2['X']
    Y = model2['Y']
    Y_pred = model2['Y_pred']
    cost = model2['cost']
    
    """
    
    n_xs = xs.shape[1]
    n_ys = ys.shape[1]
    
    #### Placeholders
    # Let's reset the graph:
    tf.reset_default_graph()

    # Create a placeholder of None x 2 dimensions (or globally n_xs) and dtype tf.float32
    # This will be the input to the network which takes the row/col
    #X = tf.placeholder(tf.float32, shape=(None, 2), name="X")
    X = tf.placeholder(tf.float32, shape=(None, n_xs), name="X")

    # Create the placeholder, Y, with 3 output dimensions (or globally n_ys) instead of 2 (or globally n_xs).
    # This will be the output of the network, the R, G, B values.
    #Y = tf.placeholder(tf.float32, shape=(None, 3), name="Y")
    Y = tf.placeholder(tf.float32, shape=(None, n_ys), name="Y")
    
    
    ############## LAYERS DEFINITION PROCESS ##################
    current_input = X
    for layer_i in range(n_layers):
        current_input = utils.linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = utils.linear(
        current_input, n_ys,
        activation=final_activation_fn,
        name='pred2')[0]
    
    
    ############## COST COMPUTATION PROCESS ##################
    # first compute the error, the inner part of the summation.
    # This should be the l1-norm or l2-norm of the distance
    # between each color channel.
    error = tf.squared_difference(Y, Y_pred)
    assert(error.get_shape().as_list() == [None, 3])
    
    # Now sum the error for each feature in Y. 
    # If Y is [Batch, Features], the sum should be [Batch]:
    sum_error = tf.reduce_sum(error, 1)
    assert(sum_error.get_shape().as_list() == [None])
    
    # Finally, compute the cost, as the mean error of the batch.
    # This should be a single value.
    cost = tf.reduce_mean(sum_error)
    assert(cost.get_shape().as_list() == [])
    
    
    
    ############## OPTIMIZATION PROCESS ##################
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Create parameters for the number of iterations to run for (< 100)
    n_iterations = n_iterations

    # And how much data is in each minibatch (< 500)
    batch_size = batch_size

    # Then create a session
    sess = tf.Session()
    
    
    ############## IMAGE PAINTING PROCESS ##################
    # Initialize all your variables and run the operation with your session
    sess.run(tf.global_variables_initializer())

    # Optimize over a few iterations, each time following the gradient
    # a little at a time
    imgs = []
    costs = []
    gif_step = gif_step
    step_i = 0

    for it_i in range(n_iterations):

        # Get a random sampling of the dataset
        idxs = np.random.permutation(range(len(xs)))

        # The number of batches we have to iterate over
        n_batches = len(idxs) // batch_size

        # Now iterate over our stochastic minibatches:
        for batch_i in range(n_batches):

            # Get just minibatch amount of data
            idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

            # And optimize, also returning the cost so we can monitor
            # how our optimization is doing.
            training_cost = sess.run(
                [cost, optimizer],
                feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]

        # Also, every 20 iterations, we'll draw the prediction of our
        # input xs, which should try to recreate our image!
        if (it_i + 1) % gif_step == 0:
            costs.append(training_cost / n_batches)
            ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
            img = np.clip(ys_pred.reshape(img.shape), 0, 1)
            imgs.append(img)
            # Plot the cost over time
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(costs)
            ax[0].set_xlabel('Iteration')
            ax[0].set_ylabel('Cost')
            ax[1].imshow(img)
            fig.suptitle('Iteration {}'.format(it_i))
            plt.show()
    
    imgs2 = imgs        
    return imgs2