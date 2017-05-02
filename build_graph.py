from __future__ import print_function
import tensorflow as tf

def build_graph():
    ###############
    # VARS
    ###############
    global_step = tf.get_variable("global_step", shape=[], trainable=False, initializer=tf.constant_initializer(1), dtype=tf.int64)
    W = {
        "W_Conv1": tf.get_variable("W_Conv1", shape=[3, 3, 1, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Conv2": tf.get_variable("W_Conv2", shape=[3, 3, 64, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Local1": tf.get_variable("W_Local1", shape=[1, 16 * 8 * 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Local2": tf.get_variable("W_Local2", shape=[1, 16 * 8 * 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine1": tf.get_variable("W_Affine1", shape=[16*8*64, 512],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine2": tf.get_variable("W_Affine2", shape=[512, 128],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "W_Affine3": tf.get_variable("W_Affine3", shape=[128, 10],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        )
    }
    b = {
        "b_Conv1": tf.get_variable("b_Conv1", shape=[1, 16, 8, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Conv2": tf.get_variable("b_Conv2", shape=[1, 16, 8, 64],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Local1": tf.get_variable("b_Local1", shape=[1, 8192],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Local2": tf.get_variable("b_Local2", shape=[1, 8192],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine1": tf.get_variable("b_Affine1", shape=[1, 512],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine2": tf.get_variable("b_Affine2", shape=[1, 128],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        ),
        "b_Affine3": tf.get_variable("b_Affine3", shape=[1, 10],
                initializer=tf.random_normal_initializer(mean=0.00, stddev=0.01),
        )}

    ###############
    # INPUTS
    ###############
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    X_input = tf.placeholder(shape=[None, 16, 8], dtype=tf.float32, name="X_input")
    y_input = tf.placeholder(shape=[None, 10], dtype=tf.int8, name="y_input")

    ###############
    # HIDDEN LAYERS #First, a very simple naive affine layer
    ###############

    #Bathch0
    inputs = tf.layers.batch_normalization(X_input)


    #Conv1
    inputs = tf.reshape(inputs, (-1, 16, 8, 1))
    inputs = tf.nn.conv2d(
                            input=inputs,
                            filter=W['W_Conv1'],
                            strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
    inputs += b['b_Conv1']
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)

    #Conv2
    inputs = tf.nn.conv2d(
                            input=inputs,
                            filter=W['W_Conv2'],
                            strides=[1, 1, 1, 1],
                            padding='SAME'
                        )
    inputs += b['b_Conv2']
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)


    #Local1
    inputs = tf.reshape(inputs, (-1, 16 * 8 * 64))
    inputs = tf.multiply(inputs, W['W_Local1'])
    inputs += b['b_Local1']
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)

    #Local2
    inputs = tf.multiply(inputs, W['W_Local2'])
    inputs += b['b_Local2']
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.dropout(inputs, keep_prob)



    #Affine1
    inputs = tf.matmul(inputs, W['W_Affine1']) + b['b_Affine1']
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.dropout(inputs, keep_prob)

    #Affine2
    inputs = tf.matmul(inputs, W['W_Affine2']) + b['b_Affine2']
    inputs = tf.layers.batch_normalization(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = tf.nn.dropout(inputs, keep_prob)

    #Affine3
    inputs = tf.matmul(inputs, W['W_Affine3']) + b['b_Affine3']
    inputs = tf.layers.batch_normalization(inputs)
    logits = tf.nn.relu(inputs)

    ###############
    # LOSS / OUTPUTS
    ###############

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
    trainer = tf.train.AdamOptimizer() #tf.train.GradientDescentOptimizer(learning_rate=learning_rate) #tf.train.AdamOptimizer()
    updateModel = trainer.minimize(loss, global_step=global_step)

    ## Accuracy
    correct_pred = tf.equal(tf.argmax(y_input, 1), tf.argmax(logits, 1))
    train_acc = tf.reduce_mean(tf.cast(correct_pred, "float"))

    return learning_rate, X_input, y_input, logits, loss, updateModel, train_acc, global_step, keep_prob
