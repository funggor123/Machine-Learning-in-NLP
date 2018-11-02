import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])

session = tf.Session()

b = session.run(a)

print(b)