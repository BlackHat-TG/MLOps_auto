#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model


# In[2]:


model = load_model('D:\htask.h5')
model.pop()


# In[5]:


for layer in model.layers:
    layer.trainable = False


# In[6]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


# In[7]:


from keras.models import Model


# In[8]:


dataset = mnist.load_data('mymnist.db')


# In[9]:


train , test = dataset


# In[10]:


X_train , y_train = train
X_test , y_test = test


# In[11]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[12]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[15]:


def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Dense(units=D, activation = "relu")(top_model)
    top_model = Dense(units= D,activation = "relu")(top_model)
    top_model = Dense(units =D, activation = "relu")(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model


# In[16]:



num_classes = 10

FC_Head = addTopModel(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())


# In[17]:


modelnew.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[18]:


h = modelnew.fit(X_train, y_train_cat, epochs=5)


# In[ ]:


final = modelnew.evaluate(x=X_test,y=y_test_cat)


# In[ ]:


accuracy=final[1]


# In[ ]:


model.save('/auto/htask.h5')
model.save_weights('/auto/h_w')


# In[ ]:


with open('/auto/acc.txt', 'w') as f:
    f.write("%f" % accuracy)

