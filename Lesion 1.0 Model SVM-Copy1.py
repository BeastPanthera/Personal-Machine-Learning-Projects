#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image


# In[2]:


from sklearn.metrics import confusion_matrix


# In[3]:


import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder


# In[4]:


from tensorflow.keras.regularizers import l2


# In[5]:


#Locating the .csv file with the data set of the skin. The data frame

skin_df_path = os.path.join('c:' + os.sep, #This part of the CODE IS INCOMPLETE. Intentionally removed location of image 
                            #data due to security purposes as this is being uploaded online.
                           )
skin_df = pd.read_csv(skin_df_path)


# In[6]:


#label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])

#Here we are taking the dx column of the array, from the skin_df array. For all of the target variables, 
#aka all the variables we will use for classification, we will have gotten each target label represented with a
#label number value. Typically labels go from 0 to n-1. For n amount of variables.


# In[7]:


#label encoding to numeric values from text
skin_df['label'] = le.transform(skin_df["dx"]) 
#This makes a new column, where the dx corressponding labels are added to the array.
print(skin_df['dx'].unique())
print(skin_df['dx'].value_counts())


# In[8]:


# Data distribution visualization
#First plot of the number of skin lesion types present in the image data.
fig1 = plt.figure()
x = ['nv','mel','bkl','bcc','akiec','vasc','df']
y = skin_df['dx'].value_counts()
c = ['red', 'pink', 'olive', 'blue', 'orange','green', 'purple']

plt.barh(x,y, color = c)
for index, value in enumerate(y):
    plt.text(value, index, str(value))
plt.ylabel('Lesion Types')
plt.xlabel('Count')
plt.title('Cancer Cell Distributions');

print(skin_df['dx'].value_counts())


# In[9]:


#Second plot of the different gender types present in the image data.
fig2 = plt.figure()

x = skin_df['sex'].unique()
y = skin_df['sex'].value_counts()
c = ['blue', 'red','purple']

plt.barh(x,y, color = c)
for index, value in enumerate(y):
    plt.text(value, index, str(value))
plt.ylabel('Genders')
plt.xlabel('Count')
plt.title('Gender Distribution of Patients');

print(skin_df['sex'].value_counts())


# In[10]:


#Third plot of the different locations of the types skin lesion types image data.
fig3 = plt.figure(figsize=(5,5))

x = ['back','lower extremity','trunk',
    'upper extremity', 'abdomen','face',
     'chest','foot','unknown','neck',
     'scalp','hand','ear','genital','acral']
y = skin_df['localization'].value_counts()
c = ['red', 'pink', 'olive', 'blue', 'orange','green', 'purple', 'grey', 
     'red', 'pink', 'olive', 'blue', 'orange','green', 'purple', 'grey',
     'red']

plt.barh(x,y, color = c)
for index, value in enumerate(y):
    plt.text(value, index, str(value))
plt.ylabel('Skin Areas')
plt.xlabel('Count')
plt.title('Skin Areas Distribution of Patients');

print(skin_df['localization'].value_counts())


# In[11]:


#Fourth plot of the different locations of the types skin lesion types image data.
fig4 = plt.figure()

#The Seaborn library is being used to help visualize the information about
#the data/ages of the patients.
x = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]
#x = [45,50,55,40,60, 70,35,65,75,30,80,85,25,20,5,15,10,0]
y = skin_df['age'].value_counts()
    
age1=skin_df['age'].dropna()
sns.distplot(age1,bins=30,kde=False)
plt.ylabel('Count')
plt.xlabel('Age')
plt.title('Age Distribution of Patients');
plt.show()
print(skin_df['age'].value_counts())


# In[12]:


#--------------------------------------
#Part 2, Resampling the data
from sklearn.utils import resample
#This is the amount of samples from each lesion type that we have.
print("This is the amount of samples from each lesion type that we have:")
print(skin_df['label'].value_counts())
print(skin_df['dx'].value_counts())


# In[13]:


#We are seperating the labelled data by their labels, making them into seperate
#DataFrames
df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]


# In[14]:


#The resample from sklearn generates extra data points of data.
#So all df databases are resampled to either upscale or downscale to 500
n_samples=500 
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42) 


# In[15]:


#Combined back to a single dataframe, with the resampled data
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                              df_2_balanced, df_3_balanced, 
                              df_4_balanced, df_5_balanced, df_6_balanced])


# In[16]:


#Check the distribution. All classes should be balanced now.
print(skin_df_balanced['label'].value_counts())


# In[17]:


#Now, we need to link each image data based on the image ID given from
#the .csv file. Will return all .jpg images. Adding it to the pandas DF.
#Below, we are specifying the path to the images.
image_pathway = os.path.join('c:' + os.sep, #This part of the CODE IS INCOMPLETE. Intentionally removed location of image 
                            #data due to security purposes as this is being uploaded online.
                            
                            )
#This creates a dictionary, the key being the image_ID, aka the name of the file.
image_path = {os.path.splitext(os.path.basename(x))[0]:x
              for x in glob(image_pathway)
             }


# In[18]:


skin_df_balanced['path'] = skin_df_balanced['image_id'].map(image_path.get)
#The .get() is recieving values from a key, aka the image_path.
#the .map() is using the .get(), to call on the skin_df_balanced['image_id'],
#using that as an input for the image_path, thus using .get() to obtain the path of the image. 

skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))
# x refers to each image path.
# Image.open(x) refers to opening the image.
# .resize((32,32) refers to resizing the image to 32 by 32.
# np.asarray(), refers to converting everything into a numpy array.
#.map() goes through all of the paths
#lambda is a short way to write a function, since we have 1 expression.


# In[19]:


n_samples = 3  # number of samples for plotting
# Plotting
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')


# In[20]:


print(skin_df['dx'].value_counts())


# In[21]:


#Convert dataframe column of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/255.0  
# Scale values to 0-1. You can also used standardscaler or other scaling methods. We need to do 
#this to normalize our data. Pixels are between 0-255.


Labels_to_images = skin_df_balanced['label']  #Assign label values to Y
#This makes a series where all the labels are corresponding with each image

Y_cat = to_categorical(Labels_to_images, num_classes=7) 
#Converting the Labels_to_images into an array that corresponds to
#if a image is a specific label, than it will equal 1 in the respective
#column, if not than it will equal 0


#Split to training and testing

x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

#Where, test_size refers to the % of the data that will be reserved for testing
#This function, train_test_split, is a function of the sklearn module.

#Where X are the feature values and Y are the label values.


# In[22]:


#PART 3: The SVM Modelling

#SVM Model:
#num_classes = 7
SIZE = 32

model2 = Sequential()
model2.add(Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3, strides = 2,input_shape=(SIZE, SIZE,3)))
model2.add(MaxPool2D(pool_size=(2,2),strides = 2))

model2.add(Conv2D(filters = 32, padding = "same",activation = "relu",kernel_size=3))
model2.add(MaxPool2D(pool_size=(2,2),strides = 2))

model2.add(Flatten())
model2.add(Dense(128,activation="relu"))

#Output layer
#model2.add(Dense(7,kernel_regularizer=l2(0.01),activation = "linear"))
model2.add(Dense(7,kernel_regularizer = l2(0.01),activation= 'softmax'))
model2.summary()


# In[23]:


#Model1:
model2.compile(loss='squared_hinge', optimizer='Adam', metrics=['accuracy'])


# In[24]:


#Model1:
batch_size = 16 
epochs = 100
#---------------------------------------------------
history_model2 = model2.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)


# In[25]:


score = model2.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


# In[26]:


#plot the training and validation accuracy and loss at each epoch
loss = history_model2.history['loss']
val_loss = history_model2.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('SVM Training & Validation Loss vs. {} Epochs'.format(len(epochs)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#The higher the loss, the less the model fits the data. 
#Training loss refers to training data, validation loss refers to new data.


# In[27]:


fig5 = plt.figure()
acc = history_model2.history['accuracy']
val_acc = history_model2.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('SVM Training & Validation Accuracy vs {} Epochs'.format(len(epochs)))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Prediction on test data
y_pred = model2.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
# Convert test data to one hot vectors
y_true = np.argmax(y_test, axis = 1) 

cf_matrix = confusion_matrix(y_true, y_pred_classes)
fig6 = plt.figure()
#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cf_matrix) / np.sum(cf_matrix, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.title('SVM Incorrect Predictions vs Labels @ {} Epochs'.format(len(epochs)))


# In[28]:


#Confusion Matrix
fig42 = plt.figure(figsize=(8,8))

group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}' for v1, v2 in
          zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(7,7)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.title('SVM Confusion Matrix @ {} Epochs'.format(len(epochs)))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')


# In[29]:


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

fpr_list = []
tpr_list = []
threshold_list = []
roc_auc_list = []

for i in range(7):
    fpr, tpr, threshold = metrics.roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc = metrics.auc(fpr, tpr)
    
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(threshold)
    roc_auc_list.append(roc_auc)

auc_value = []
for i in range(7):
    auc_value.append(auc(fpr_list[i],tpr_list[i]))
fig56 = plt.figure(figsize=(7,7))
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr_list[0], tpr_list[0], label="akiec Label ROC, Area curve of {:.4f}".format(auc_value[0]))
plt.plot(fpr_list[1], tpr_list[1], label="bcc Label ROC, Area curve of {:.4f}".format(auc_value[1]))
plt.plot(fpr_list[2], tpr_list[2], label="bkl label ROC, Area curve of {:.4f}".format(auc_value[2]))
plt.plot(fpr_list[3], tpr_list[3], label="df label ROC, Area curve of {:.4f}".format(auc_value[3]))
plt.plot(fpr_list[4], tpr_list[4], label="mel label ROC, Area curve of {:.4f}".format(auc_value[4]))
plt.plot(fpr_list[5], tpr_list[5], label="nv label ROC, Area curve of {:.4f}".format(auc_value[5]))
plt.plot(fpr_list[6], tpr_list[6], label="vasc label ROC, Area curve of {:.4f}".format(auc_value[6]))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC & AUC Curve @ {} Epochs".format(len(epochs)))
plt.legend(loc="lower right")
plt.show()


# In[ ]:




