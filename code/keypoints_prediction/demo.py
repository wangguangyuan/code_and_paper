import tensorflow as tf
import numpy as np
labels=np.array([[0],[2]])
print(labels.shape)
input_data = tf.Variable([[0.2, 0.1, 0.9], [0.3, 0.4, 0.6]], dtype=tf.float32)
output = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=input_data, labels=labels)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('softmax_cross:',sess.run(output))


