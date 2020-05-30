#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model


# In[2]:


dataset = mnist.load_data('mymnist.db')


# In[3]:


train , test = dataset


# In[4]:


X_train , y_train = train
X_test , y_test = test


# In[5]:


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)
X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[6]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[7]:


model = Sequential()


# In[8]:


model.add(Dense(units=1024, input_dim=28*28, activation='relu'))
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


# In[9]:


model.summary()


# In[10]:


model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[12]:


h = model.fit(X_train, y_train_cat, epochs=5)


# In[ ]:


final = model.evaluate(x=X_test,y=y_test_cat)


# In[ ]:


accuracy=final[1]


# In[ ]:


model.save('/auto/htask.h5')
model.save_weights('/auto/h_w')


# In[ ]:


with open('/auto/acc.txt', 'w') as f:
    f.write("%f" % accuracy)


# In[ ]:




