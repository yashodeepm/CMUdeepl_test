
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
from tensorflow.keras import optimizers
from shutil import copyfile
# In[2]:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class myCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        tf.keras.callbacks.Callback.__init__(self)
        self.prev_log = 10000000
        #print(self.prev_log)
    def on_epoch_begin(self, epoch, logs = {}):
        self.prev_log = logs.get('loss')
        if(self.prev_log==None):
            self.prev_log = 1000000
#        print(self.prev_log)
    def on_epoch_end(self, epoch, logs = {}):
#        print(logs.keys())
        if(logs.get('loss')!=None and logs.get('loss')<0.0000001):
            self.model.stop_training = True
callbacks = myCallback()

# In[10]:

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), input_shape = (480, 640, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
#    tf.keras.layers.Dense(1024),
#    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(480*640*3)
])
sgd = optimizers.SGD(lr = 0.1, decay = 1e-5)
model.compile(optimizer = sgd, loss = 'mean_squared_error')
model.summary()


# In[4]:
#os.mkdir('test')
#os.chdir('test')
#os.mkdir('x_train')
#os.mkdir('y_train')
#os.chdir('x_train')
#base = '/home/ymahapatra/Set1/'
#prediction_time = 30
#for i in os.listdir(base):
#    os.chdir(base+i)
#    iterator_list = sorted(os.listdir())
#    for j in range(len(iterator_list)-prediction_time):
#        copyfile(iterator_list[j], '/home/ymahapatra/test/x_train/'+iterator_list[j])
#        copyfile(iterator_list[j+prediction_time], '/home/ymahapatra/test/y_train/'+iterator_list[j+prediction_time])

checkpoint_path = "/home/ymahapatra/trained_models/predictor/cp_30s.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True)

test_data_size = 1
x_train = []
y_train = []
os.chdir('/home/ymahapatra/test')

for i in sorted(os.listdir('x_train'))[0:test_data_size]:
#    print('x_train/'+i)
    x_train.append(cv2.imread('x_train/'+i, 1))
for i in sorted(os.listdir('y_train'))[0:test_data_size]:
#    print('y_train/'+i)
    y_train.append(cv2.imread('y_train/'+i, 1))


# In[5]:

x_train = np.array(x_train, dtype = float)
y_train = np.array(y_train, dtype = float)
x_train, y_train = x_train/255, y_train/255
#print(x_train.shape)
#print(y_train.shape)
y_train = y_train.reshape(-1, 480*640*3)
#print(y_train.shape)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
model.load_weights(latest)

# In[8]:
model.fit(x_train, y_train, epochs = 1, callbacks = [cp_callback])
#loss = model.evaluate(x_train[0].reshape(1,480,640,3), y_train[0].reshape(1,921600))
#print(loss)
#print(x_train[0])
#print(y_train[0])
#cmd = 'zip -r t.zip trained_models'
#os.system(cmd)
# In[45]:




# In[ ]:



