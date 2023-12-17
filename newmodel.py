#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


# In[2]:


def load_and_preprocess_data(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0   # normalize pixel values between 0 and 1
        images.append(img)
        labels.append(label)
    return images, labels


# In[4]:


angry_images, angry_labels = load_and_preprocess_data('dataset/angry', label=0)
fear_images, fear_labels = load_and_preprocess_data('dataset/fear', label=1)
happy_images, happy_labels = load_and_preprocess_data('dataset/happy', label=2)
neutral_images, neutral_labels = load_and_preprocess_data('dataset/neutral', label=3)
sad_images, sad_labels = load_and_preprocess_data('dataset/sad', label=4)
surprise_images, surprise_labels = load_and_preprocess_data('dataset/surprise', label=5)


# In[5]:


all_images = np.concatenate([angry_images, fear_images, happy_images, neutral_images, sad_images, surprise_images])
all_labels = np.concatenate([angry_labels, fear_labels, happy_labels, neutral_labels, sad_labels, surprise_labels])


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)


# In[7]:


# Load the ResNet50 model with pre-trained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
# Add your custom dense layers on top of the pre-trained model
res_model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')  # Assuming you have 6 classes
])


# In[8]:


batch_size = 64
nb_epochs = 20


# In[9]:


# Add callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)


# In[11]:


# Compile and train the model with callbacks
res_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
res_model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1,
              validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])


# In[12]:


score = res_model.evaluate(X_test, y_test,verbose=0 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])


# In[13]:


res_model.save("newmodel_res.h5")


# In[14]:


from tensorflow.keras.models import load_model
loaded_model=load_model("newmodel_res.h5")


# In[15]:


# Function to preprocess a test image
def preprocess_test_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        return img
    else:
        return None


# In[17]:


class_labels = {
        0: 'anger',
        1: 'fear',
        2: 'happy',
        3: 'neutral',
        4: 'sad',
        5: 'surprise'
    }


# In[18]:







# In[ ]:




