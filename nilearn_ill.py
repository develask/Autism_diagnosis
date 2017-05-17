import os
import csv
import numpy as np
import nibabel as nib


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D

#from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD

#from keras.utils import np_utils
#from keras import backend as K
import tensorflow as tf


directory = "./ABIDE/weighted_degree"
nombres = []
images = []
for filename in os.listdir(directory):
    if filename.endswith(".nii.gz"): 
        images.append(nib.load(os.path.join(directory, filename)))
        nombres.append(filename)



x_data = []
for image in images:
    x_data.append(image.get_data())

x_data = np.asarray(x_data)

print("Shape of input:")
print(x_data.shape)

x_data = x_data[:,:10,:10,:10]
print(x_data.shape)


y_data = np.full((x_data.shape[0],1), 10, dtype = np.dtype('int16'))

with open('./ABIDE/Phenotypic_V1_0b.csv') as csvfile:
    d = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
    for row in d:
        name = row[0].title()+"_00"+row[1]+".nii.gz"
        try:
            i = nombres.index(name)
            y_data[i] = row[3]
        except Exception as e:
            pass
print("Shape of output:")    
print(y_data.shape)




to_select = y_data != 10
to_select = to_select.reshape(to_select.shape[0])
print(to_select.shape)
print(x_data.shape)
x_data = x_data[to_select,:,:,:]
y_data = y_data[to_select.T,:]
y_data = y_data > 0

x_data = x_data.reshape(x_data.shape[0],x_data.shape[1],x_data.shape[2],x_data.shape[3],1)
#create output for the net
y_data = np.concatenate((y_data, 1-y_data), axis=1)

print("New shapes:")
print(x_data.shape)
print(y_data.shape)
print("brains with no lesion:", np.sum(y_data[:,0] == 0))






model = Sequential()
input_shape=x_data.shape[1:]

# number of convolutional filters to use
nb_filters = 8
# size of pooling area for max pooling
pool_size = (2, 2, 2)
# convolution kernel size
kernel_size = (3, 3, 3)

nb_classes = 2
print("Input shape to the network:", input_shape)
model.add(Convolution3D(nb_filters, kernel_size[0], kernel_size[1], kernel_size[2],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
print("Output shape of 1st convolution:", model.output_shape)

model.add(MaxPooling3D(pool_size=pool_size))
print("Output shape after a max pooling:", model.output_shape)

model.add(Convolution3D(nb_filters, kernel_size[0], kernel_size[1], kernel_size[2]))
model.add(Activation('relu'))
print("Output shape of 2nd convolution:", model.output_shape)

#model.add(MaxPooling3D(pool_size=pool_size))
#print("Output shape after a max pooling:", model.output_shape)

#model.add(Convolution3D(nb_filters, kernel_size[0], kernel_size[1], kernel_size[2]))
#model.add(Activation('relu'))
#print("Output shape of 3rd convolution:", model.output_shape)

#model.add(MaxPooling3D(pool_size=pool_size))
#print("Output shape after a max pooling:", model.output_shape)

model.add(Flatten())
print("Output shape after flatten:", model.output_shape)

model.add(Dense(128))
print("Output shape after (dense 128):", model.output_shape)
model.add(Activation('relu'))
print("Output shape after activation(relu):", model.output_shape)

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
print("Output shape after softmax (2 classes):", model.output_shape)




batch_size = 128
nb_epoch = 100

init_ler = 0.05
final_ler = 0.005
dec = (final_ler/init_ler)**(1/nb_epoch)

sgd = SGD(lr=init_ler,decay=dec,momentum=0.9,nesterov = False)
model.compile(loss='binary_crossentropy',
                   optimizer=sgd,
                   metrics=['accuracy'])
tr_h = model.fit(x_data, y_data, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1)
