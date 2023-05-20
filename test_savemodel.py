from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import mnist
from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)/255.0
x_test = x_test.reshape(x_test.shape[0], -1)/255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

inputs = Input(shape=(784, ))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
y = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=y)

# model.save('m1.h5')
model.save_weights('m3')
# model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=32, epochs=10)
# #loss,accuracy=model.evaluate(x_test,y_test)

# model.save('m2.h5')
# model.save_weights('m3.h5')
