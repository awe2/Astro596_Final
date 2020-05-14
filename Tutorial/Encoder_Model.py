
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.utils import Sequence


# In[2]:
BATCH_SIZE=64

NB_BINS = 60 * 3
ZMIN = 0.0
ZMAX = 0.4
BIN_SIZE = (ZMAX - ZMIN) / NB_BINS
range_z = np.linspace(ZMIN, ZMAX, NB_BINS + 1)[:NB_BINS]

ID_list = os.listdir('./Spectra')
ID_list = [sub[ : -4] for sub in ID_list]
ID_list = np.array(ID_list,dtype=np.int64)

I_got = pd.DataFrame()
I_got['specObjID']=ID_list
# In[3]:

tf.config.experimental.set_visible_devices([], 'GPU')

cols=['specObjID','z','dered_u','dered_g','dered_r','dered_i','dered_z']

X = pd.read_csv('../Photo_Z/DATA/SDSS_DR12_awe2_0.csv',usecols=cols)

X.drop_duplicates(subset=['specObjID'],inplace=True)

X = pd.merge(left=X,right=I_got,left_on='specObjID',right_on='specObjID',how='inner')

#Labels1 is a dictionary of spectraID to Z

#labels2 is a dictionary of spectraID to an array np.array([u,g,r,i,z])

redshift_dictionary = dict(zip((X['specObjID'].values).tolist(),(X['z'].values).tolist()))

magnitude_dictionary = dict(zip((X['specObjID'].values).tolist(),(X[['dered_u','dered_g','dered_r','dered_i','dered_z']].values).tolist()))

X = X[X['z']<=0.4]
X.reset_index(inplace=True)
Y = X['z'].values

train_IDs = X['specObjID'].iloc[0:175000].values
val_IDs = X['specObjID'].iloc[175000:300000].values
test_IDs = X['specObjID'].iloc[300000::].values
partition = {'train':train_IDs,'validation':val_IDs,'test':test_IDs}

NOW = X.drop(columns=['specObjID','z','index'])

def make_cat(Y):
    return tf.keras.utils.to_categorical(np.round((180-1)*(Y/0.4),0).astype(int))

#grab a reserved testing set, of 300000 values
x_train=NOW.iloc[0:175000].values
y_train=make_cat(Y[0:175000])

x_val=NOW.iloc[175000:300000].values
y_val=make_cat(Y[175000:300000])

x_test = NOW.iloc[300000::].values
y_test= make_cat(Y[300000::])


# In[4]:


class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels1, labels2, batch_size=64, dim=(3800), n_channels=1,
                 n_classes=2, shuffle=True):
     #   'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels1 = labels1
        self.labels2 = labels2
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
    #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, 1), dtype=np.float32)
        y1 = np.empty((self.batch_size,5))
        y2 = np.empty((self.batch_size))
        # Generate data and perform augmentation
        for i,ID in enumerate(list_IDs_temp):
            #print(list_IDs_temp)
            # Store sample
            a = np.load('./Spectra/' + str(ID) + '.npy')
            if len(a) > 3800:
                X[i,:,:] = a[0:3800,np.newaxis]
            else:
                X[i,0:len(a),:] = a[:,np.newaxis]
            
            #store magnitude target
            y1[i,] = self.labels1[ID]
            #store spectro target
            y2[i,] = self.labels2[ID]
        
        if self.n_classes > 2:
            y2 = (np.round((y2/0.4) * (self.n_classes-1),0)).astype(int) #noramlize, bin
            #y[self.n_channels<=y] = self.n_classes-1 I think this is fucking with me, should already be taken care of below
            #y = keras.utils.to_categorical(y, num_classes=self.n_classes)
            return (X, [y1,keras.utils.to_categorical(y2, num_classes=self.n_classes)])
        else:
            return (X, [y1,y2])
    
    def __len__(self):
    #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
    #  'Generate one batch of data'
      # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

      # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

      # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y


# In[5]:




          
params_train = {'dim': (3800,),
          'batch_size': BATCH_SIZE,
          'n_classes': 180,
          'n_channels': 1,
          'shuffle': True}


# In[11]:


train_generator = DataGenerator(list_IDs=partition['train'], labels1=magnitude_dictionary, labels2=redshift_dictionary, **params_train)
train_steps_to_take = int(len(train_IDs)/BATCH_SIZE)

 
def encoder_model():
    Input = keras.layers.Input((3800,1),)
    conv1 = keras.layers.Conv1D(filters=64,kernel_size=9,dilation_rate=1)(Input)
    pool1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    conv2 = keras.layers.Conv1D(filters=64,kernel_size=5,dilation_rate=1)(conv1)
    pool2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    flat = keras.layers.Flatten()(pool2)
    edense1 = keras.layers.Dense(252,activation=keras.activations.relu)(flat)
    edense2 = keras.layers.Dense(128,activation=keras.activations.relu)(edense1)
    edense3 = keras.layers.Dense(5,activation=keras.activations.linear,name='output1')(edense2)
    #Pairwise_Subtraction()(edense3)
    dense1 = keras.layers.Dense(45,activation=keras.activations.relu)(edense3)
    drop1 = keras.layers.Dropout(0.05)(dense1)
    
    dense2 = keras.layers.Dense(45,activation=keras.activations.relu)(drop1)
    drop2 = keras.layers.Dropout(0.05)(dense2) #0.1
    
    dense3 = keras.layers.Dense(45,activation=keras.activations.relu)(drop2)
    drop3 = keras.layers.Dropout(0.05)(dense3)
    
    dense4 = keras.layers.Dense(45,activation=keras.activations.relu)(drop3)
    drop4 = keras.layers.Dropout(0.05)(dense4)#0.1
    
    dense5 = keras.layers.Dense(180,activation=keras.activations.softmax,name='output2')(drop4)
    
    model = keras.Model(inputs=[Input],outputs=[edense3,dense5])
    return model

def MLP():
    IN = keras.layers.Input((5,))
    dense1 = keras.layers.Dense(45,activation=keras.activations.relu)(IN)
    drop1 = keras.layers.Dropout(0.05)(dense1)
    
    dense2 = keras.layers.Dense(45,activation=keras.activations.relu)(drop1)
    drop2 = keras.layers.Dropout(0.05)(dense2) #0.1
    
    dense3 = keras.layers.Dense(45,activation=keras.activations.relu)(drop2)
    drop3 = keras.layers.Dropout(0.05)(dense3)
    
    dense4 = keras.layers.Dense(45,activation=keras.activations.relu)(drop3)
    drop4 = keras.layers.Dropout(0.05)(dense4)#0.1
    
    dense5 = keras.layers.Dense(180,activation=keras.activations.softmax)(drop4)
    
    model = keras.Model(inputs=[IN],outputs=[dense5])
    return(model)


# In[14]:


encoder = encoder_model()
adam_e = tf.keras.optimizers.Adam(lr=1e-4)#5e-4 unstable, reduce

losses = {
    "output1": "MSE",
    "output2": "categorical_crossentropy",
}

lossWeights = {"output1": 1.0, "output2": 1.0}

encoder.compile(optimizer=adam_e, loss=losses, loss_weights=lossWeights)

predictor = MLP()
adam_p = tf.keras.optimizers.Adam(lr=1e-4)


predictor.compile(optimizer=adam_p,loss='categorical_crossentropy')

#compare s=0.1 to MSE; fucked up it was (2*0.1)**2 istead of 2*(0.1)**2.account for that if it matters to us in an hour

#Need to find where my guy overtrains and strike a balance


# In[20]:


filepath='predictor_weights.hdf5'

best_loss = 1000
print('how many training setps? ', train_steps_to_take)
print('got to the training loop.')
for i in range(1,30):
    print('training loop ',i)
    encoder.fit_generator(generator=train_generator,
                       steps_per_epoch=train_steps_to_take,
                       epochs=1,
                       initial_epoch=0,
                       verbose=1)
    
    weights1 = encoder.get_weights()
    predictor.set_weights(weights1[10::])
    
    new_loss = predictor.evaluate(x=x_val,y=y_val,verbose=1)
    if new_loss < best_loss:
        print('best loss beaten, saving')
        predictor.save_weights(filepath)
        best_loss=new_loss

