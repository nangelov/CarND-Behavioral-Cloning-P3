import csv
import cv2
import numpy as np


lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction = 0.5  # this is a parameter to tune

for line in lines:
	#center
	source_path = line[0]
	filename = source_path.split('/')[-1]
	image_center = cv2.imread(filename)
	image_center = cv2.GaussianBlur(image_center,(5,5),0)
	images.append(image_center)
	measurement = float(line[3])
	measurements.append(measurement)
	#flip
	image_flipped = np.fliplr(image_center)
	image_flipped = cv2.GaussianBlur(image_flipped,(5,5),0)
	images.append(image_flipped)
	measurement_flipped = -measurement
	measurements.append(measurement_flipped)
	#left
	source_path = line[1]
	filename = source_path.split('/')[-1]
	image_left = cv2.imread(filename)
	image_left = cv2.GaussianBlur(image_left,(5,5),0)
	images.append(image_left)
	steering_left = measurement + correction
	measurements.append(steering_left)
	# flip
	image_flipped = np.fliplr(image_left)
	image_flipped = cv2.GaussianBlur(image_flipped,(5,5),0)
	images.append(image_flipped)
	measurement_flipped = -steering_left
	measurements.append(measurement_flipped)
	#right
	source_path = line[2]
	filename = source_path.split('/')[-1]
	image_right = cv2.imread(filename)
	image_right = cv2.GaussianBlur(image_right,(5,5),0)
	images.append(image_right)
	steering_right = measurement - correction
	measurements.append(steering_right)
	# flip
	image_flipped = np.fliplr(image_right)
	image_flipped = cv2.GaussianBlur(image_flipped,(5,5),0)
	images.append(image_flipped)
	measurement_flipped = -steering_right
	measurements.append(measurement_flipped)

print("Dataet prepared!")

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, GaussianDropout

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(GaussianDropout(2))
model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)


model.save('model.h5')

print("model saved!")

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()