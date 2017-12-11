import h5py
import numpy as np
import random
import tensorflow as tf
from scipy.misc import imsave

class Data:
  def __init__(self):
    with h5py.File("cell_data.h5", "r") as data:
      self.train_images = [data["/train_image_{}".format(i)][:] for i in range(28)]
      self.train_labels = [data["/train_label_{}".format(i)][:] for i in range(28)]
      self.test_images = [data["/test_image_{}".format(i)][:] for i in range(3)]
      self.test_labels = [data["/test_label_{}".format(i)][:] for i in range(3)]
    
    self.input_resolution = 300
    self.label_resolution = 116

    self.offset = (300 - 116) // 2

  def get_train_image_list_and_label_list(self):
    n = random.randint(0, len(self.train_images) - 1)
    x = random.randint(0, (self.train_images[n].shape)[1] - self.input_resolution - 1)
    y = random.randint(0, (self.train_images[n].shape)[0] - self.input_resolution - 1)
    image = self.train_images[n][y:y + self.input_resolution, x:x + self.input_resolution, :]

    x += self.offset
    y += self.offset
    label = self.train_labels[n][y:y + self.label_resolution, x:x + self.label_resolution]
    
    return [image], [label]

  def get_test_image_list_and_label_list(self):
    coord_list = [[0,0], [0, 116], [0, 232], 
                  [116,0], [116, 116], [116, 232],
                  [219,0], [219, 116], [219, 232]]
    
    image_list = []
    label_list = []
    
    for image_id in range(3):
      for y, x in coord_list:
        image = self.test_images[image_id][y:y + self.input_resolution, x:x + self.input_resolution, :]
        image_list.append(image)
        x += self.offset
        y += self.offset
        label = self.test_labels[image_id][y:y + self.label_resolution, x:x + self.label_resolution]
        label_list.append(label)
    

    return image_list, label_list

###############################################################################

def intersection_over_union(pred, labels):
	pred = pred[:,:,:,0] >= 0.5
	labels_c = np.array(labels).astype(bool)
	correct = np.zeros(pred.shape[0])
	sample = 0
	while sample < pred.shape[0]:
		y = 0
		while y < pred.shape[1]:
			x = 0
			while x < pred.shape[2]:
				correct[sample] += labels_c[sample][y][x] and pred[sample][y][x]
				x += 1
			y += 1
		sample += 1
	total = np.zeros(pred.shape[0])
	sample = 0
	while sample < pred.shape[0]:
		y = 0
		while y < pred.shape[1]:
			x = 0
			while x < pred.shape[2]:
				total[sample] += labels_c[sample][y][x]
				x += 1
			y += 1
		sample += 1
	incorrect = np.zeros(pred.shape[0])
	sample = 0
	while sample < pred.shape[0]:
		y = 0
		while y < pred.shape[1]:
			x = 0
			while x < pred.shape[2]:
				incorrect[sample] += labels_c[sample][y][x] < pred[sample][y][x]
				x += 1
			y += 1
		sample += 1
	return np.mean(correct / (total + incorrect + 0.000001))

def conv2d(inputs, filters, kernel_size, strides=(1,1), activation=tf.nn.relu):
	return tf.layers.conv2d(
		inputs=inputs, filters=filters, kernel_size=kernel_size,
		strides=strides, padding='valid', activation=activation,
		bias_initializer=tf.zeros_initializer(),
		kernel_initializer=tf.glorot_uniform_initializer())

def conv2d_transpose(inputs, filters, kernel_size, strides):
	return tf.layers.conv2d_transpose(
		inputs=inputs, filters=filters, kernel_size=kernel_size,
		strides=strides, padding='valid')

def max_pool2d(inputs, pool_size, strides):
	return tf.layers.max_pooling2d(
		inputs=inputs, pool_size=pool_size, strides=strides)

def crop_concat(left, right):
	""" Left needs to be the larger image. """
	left_height, left_width = left.get_shape().as_list()[1:3]
	right_height, right_width = right.get_shape().as_list()[1:3]
	offset_height = (left_height - right_height) // 2
	offset_width = (left_width - right_width) // 2
	left_cropped = tf.image.crop_to_bounding_box(left,
		offset_height, offset_width, right_height, right_width)
	concatenated = tf.concat([left_cropped, right], -1)
	return concatenated

x = tf.placeholder(tf.float32, shape=[None, 300, 300, 1])
labels_input = tf.placeholder(tf.float32, shape=[None, 116, 116])
labels_reshaped = tf.reshape(labels_input, (-1, 116, 116, 1))
y_ = tf.concat([labels_reshaped, 1.0 - labels_reshaped], -1)
convolution_1 = conv2d(x, 32, (3, 3))
convolution_2 = conv2d(convolution_1, 32, (3, 3))
max_pool_3 = max_pool2d(convolution_2, (2, 2), (2, 2))
convolution_4 = conv2d(max_pool_3, 64, (3, 3))
convolution_5 = conv2d(convolution_4, 64, (3, 3))
max_pool_6 = max_pool2d(convolution_5, (2, 2), (2, 2))
convolution_7 = conv2d(max_pool_6, 128, (3, 3))
convolution_8 = conv2d(convolution_7, 128, (3, 3))
max_pool_9 = max_pool2d(convolution_8, (2, 2), (2, 2))
convolution_10 = conv2d(max_pool_9, 256, (3, 3))
convolution_11 = conv2d(convolution_10, 256, (3, 3))
max_pool_12 = max_pool2d(convolution_11, (2, 2), (2, 2))
convolution_13 = conv2d(max_pool_12, 512, (3, 3))
convolution_14 = conv2d(convolution_13, 512, (3, 3))
c_transpose_15 = conv2d_transpose(convolution_14, 256, (2, 2), (2, 2))
concatenated_16 = crop_concat(convolution_11, c_transpose_15)
convolution_17 = conv2d(concatenated_16, 256, (3, 3))
convolution_18 = conv2d(convolution_17, 256, (3, 3))
c_transpose_19 = conv2d_transpose(convolution_18, 128, (2, 2), (2, 2))
concatenated_20 = crop_concat(convolution_8, c_transpose_19)
convolution_21 = conv2d(concatenated_20, 128, (3, 3))
convolution_22 = conv2d(convolution_21, 128, (3, 3))
c_transpose_23 = conv2d_transpose(convolution_22, 64, (2, 2), (2, 2))
concatenated_24 = crop_concat(convolution_5, c_transpose_23)
convolution_25 = conv2d(concatenated_24, 64, (3, 3))
convolution_26 = conv2d(convolution_25, 64, (3, 3))
c_transpose_27 = conv2d_transpose(convolution_26, 32, (2, 2), (2, 2))
concatenated_28 = crop_concat(convolution_2, c_transpose_27)
convolution_29 = conv2d(concatenated_28, 32, (3, 3))
convolution_30 = conv2d(convolution_29, 32, (3, 3))
convolution_31 = conv2d(convolution_30, 2, (1, 1), activation=None)

# Need a seperate softmax layer for visualisation.
softmax = tf.nn.softmax(convolution_31)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,
	logits=convolution_31))
optimiser = tf.train.AdamOptimizer(0.0001, 0.95, 0.99)
train_step = optimiser.minimize(loss)

data = Data()
v_imgs, v_labels = data.get_test_image_list_and_label_list()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(40000):
		b_imgs, b_labels = data.get_train_image_list_and_label_list()
		train_step.run(feed_dict={x: b_imgs, labels_input: b_labels})
		if i % 100 == 0:
			t_pred = softmax.eval(feed_dict={x: b_imgs, labels_input: b_labels})
			t_acc = intersection_over_union(t_pred, b_labels)
			v_pred = softmax.eval(feed_dict={x: v_imgs, labels_input: v_labels})
			v_acc = intersection_over_union(v_pred, v_labels)
			print('{}\t{:.6f}\t{:.6f}'.format(i, t_acc, v_acc))

	v_pred = softmax.eval(feed_dict={x: v_imgs, labels_input: v_labels})
	v_pred = (v_pred[:,:,:,0] >= 0.5).astype(float)
	for j in range(v_pred.shape[0]):
		imsave('{}_img.png'.format(j), np.reshape(v_imgs[j], (300,300))[92:208,92:208])
		imsave('{}_label.png'.format(j), np.reshape(v_labels[j], (116,116)))
		imsave('{}_pred.png'.format(j), np.reshape(v_pred[j], (116,116)))
