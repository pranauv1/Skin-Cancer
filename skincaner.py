###### I've used Google Colab to do the training (Some lines may differ when training locally) ######

"""
Skin cancer lesion classification using the HAM10000 dataset

Dataset link:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

"""

### Google Colab ###
#To Download Data
! kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

#Unzip files
!unzip skin-cancer-mnist-ham10000.zip
### Google Colab ###


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

SIZE=32
np.random.seed(42)


#Read CSV file
skin_df = pd.read_csv('/path/HAM10000_metadata.csv')

#Read images based on ID from CSV
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('/path/', '*', '*.jpg'))}

#Add the image path(s) to datafram(skin_df) as a new column 
skin_df['path'] = skin_df['image_id'].map(image_path.get)

#Use the path(s) to read image(s) --> resize images into 32X32 --> convert images into np.array --> add them into a new column
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))

##### Plotting #####
n_samples = 5
# Plotting
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
##### Plotting #####
        
        
#Label encoding --> From text to numeric values
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
#Transform and Add those labels to dataframe(skin_df) as a new column
skin_df['label']=le.transform(skin_df['dx'])


#######Plotting#########
#Data distribution visualization
#We're just looking at Cell Type(cancer type) since that's what we're gonna deal with
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type');

plt.tight_layout()
plt.show()
#######Plotting#########


#Balancing data.

#Get labels and counts --> Assign them into a new DataFrames
df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]

#Resamplling those DataFrames
n_samples = 1000
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

#Combining those DataFrames to a new Single DataFrame
skin_df_balanced = pd.concat([df_0_balanced,df_1_balanced,df_2_balanced,df_3_balanced,df_4_balanced,df_5_balanced,df_6_balanced])

#Will check balanced classes
print(skin_df_balanced['label'].value_counts())




#Creating the X and Y for Training and Testing

#Converting 'image(s)' from Dataframe(skin_df_balanced) to np.array
X = np.asarray(skin_df_balanced['image'].tolist())
#Sclling those values from 0-255 to 0-1.
X=X/255.

#Assigning 'label(s)' from Dataframe(skin_df_balanced) to Y
Y=skin_df_balanced['label']
#Since this a multiclass problem we need to conver those Y values into 'categorical'
Y_cat = to_categorical(Y, num_classes=7)

#Will Split Training and Testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)


############# Model #################

#Finally, will define the model
num_calasses = 7

model = Sequential()
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
#BatchNormalization
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation="relu"))
#BatchNormalization
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation="relu"))
#BatchNormalization
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["acc"])

############# Model #################

#Let's Train

batch_size = 16
epochs = 85

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)


#The Final Score
score = model.evaluate(x_test, y_test)
print('Test Accuracy:', score[1]*100, '%')


######### Plotting #########
#plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
######### Plotting #########


######### Plotting #########
#plot the training and validation accuracy at each epoch
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
######### Plotting #########


######### Plotting #########
#Plot fractional incorrect misclassifications
cm = confusion_matrix(y_true, y_pred_classes)
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
######### Plotting #########

#Saving the model
model.save('skinCancer.h5')
