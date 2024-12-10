import os
#uncomment this line and skip gpu if running issues with model training in gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
import time
import datetime
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
sns.set(font_scale=2,style='whitegrid')
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold


#set random seed across different library/components
seed_val = 42
np.random.seed(seed=seed_val)
try:
    tf.random.set_random_seed(seed_val)
except:
    tf.random.set_seed(seed_val)
os.environ['PYTHONHASHSEED']=str(seed_val)
random.seed(seed_val)

# Index for each activity
activity_indices = {
  'stationary': 0,
  'gesture 1': 1,
  'gesture 2': 2,
  }


standard_scaler = StandardScaler()
minmax_scaler   = MinMaxScaler()

def compute_raw_data(dir_name):
  '''
  Given a directory location, this function returns the raw data and activity labels
  for data in that directory location
  '''
  raw_data_features = None
  raw_data_labels = None
  interpolated_timestamps = None

  sessions = set()
  # Categorize files containing different sensor sensor data
  file_dict = dict()
  # List of different activity names
  #activities = set()
  file_names = os.listdir(dir_name)
  #print(file_names)
  for file_name in file_names:
    if '.txt' in file_name:

      activity = dir;
      name = file_name[:-4]
      sessions.add((activity,name))

      if (activity, name) in file_dict:
        file_dict[(activity,name)].append(file_name)
      else:
        file_dict[(activity,name)] = [file_name]

  #print(sessions)
  for session in sessions:
    file = file_dict[session[0],session[1]][0]
    # print(file)

    #TODO: add gyroscope data

    sensor_df = pd.read_csv(dir_name + '/' + file, names =['time', 'accx','accy','accz', 'gx', 'gy', 'gz'])

    #sensor = sensor_df.drop_duplicates(sensor_df.columns[0], keep='first').values

    sensor_df['time']=sensor_df['time'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f"))

    sensor_df['time']=(sensor_df['time']-datetime.datetime(1970,1,1))
    sensor_df['time']=sensor_df['time'].dt.total_seconds()

    #print(time.mktime(datetime.datetime.strptime(sensor[-1, 0], "%Y-%m-%d %H:%M:%S.%f").timetuple()))

    # Spine-line interpolataion for x, y, z values (sampling rate is 32Hz).
    # Remove data in the first and last 3 seconds.
    #timestamps in the file are in milliseconds

    timestamps = np.arange(sensor_df['time'].iloc[0],sensor_df['time'].iloc[-1],8/1000)


    sensor_ip = np.stack([np.interp(timestamps, sensor_df['time'], sensor_df['accx']),
                      np.interp(timestamps, sensor_df['time'], sensor_df['accy']),
                      np.interp(timestamps, sensor_df['time'], sensor_df['accz']),
                      np.interp(timestamps, sensor_df['time'], sensor_df['gx']),
                      np.interp(timestamps, sensor_df['time'], sensor_df['gy']),
                      np.interp(timestamps, sensor_df['time'], sensor_df['gz'])],
                     axis=1)

    #print(sensor_ip)
    # print("hello")
    # Keep data with dimension multiple of 8
    length_multiple_8 = 8*int(sensor_ip.shape[0]/8)
    sensor_ip = sensor_ip[0:length_multiple_8, :]
    # print(session)
    # print(activity_indices)
    # print(sensor_ip.shape[0]*[int(activity_indices)])

    labels = np.array(sensor_ip.shape[0]*[int(activity_indices[session[0]])]).reshape(-1, 1)
    timestamps = timestamps[0:length_multiple_8]
    print(len(sensor_ip))
    print(len(labels))
    print(len(timestamps))

    if raw_data_features is None:
      raw_data_features = sensor_ip
      raw_data_labels = labels
      interpolated_timestamps = timestamps
    else:
      raw_data_features = np.append(raw_data_features, sensor_ip, axis=0)
      raw_data_labels = np.append(raw_data_labels, labels, axis=0)
      interpolated_timestamps = np.append(interpolated_timestamps, timestamps, axis=0)

  return raw_data_features, raw_data_labels, interpolated_timestamps

def feature_extraction_lstm_raw(raw_data_features, raw_data_labels, timestamps):
  """
  Takes in the raw data and labels information and returns data formatted
  for processing in a lstm model (batch, timesteps, features) format
  Args:
    raw_data_features: raw data returns from the directory The fourth column is the barometer data.
    raw_data_labels: labels associated with a data row
    timestamps: timestamp of the given row of observation
  Returns:
    features_np: features (re-arrange raw data or other derived observation) according to lstm format
    labels_np: labels associated with each of the features
  """
  features = []
  labels   = []

  #accel_magnitudes = butter_lowpass_filter(accel_magnitudes,4,32)
  accel_magnitudes = np.sqrt((raw_data_features[:, 0]**2).reshape(-1, 1)+
                             (raw_data_features[:, 1]**2).reshape(-1, 1)+
                             (raw_data_features[:, 2]**2).reshape(-1, 1))
  gyro_magnitudes = np.sqrt((raw_data_features[:, 3] ** 2).reshape(-1, 1) +
                             (raw_data_features[:, 4] ** 2).reshape(-1, 1) +
                             (raw_data_features[:, 5] ** 2).reshape(-1, 1))

  # The window size for feature extraction
  segment_size = 16

  for i in range(0, raw_data_features.shape[0]-segment_size, segment_size):
    segment    = raw_data_features[i:i+segment_size]
    seg_accmag = accel_magnitudes[i:i+segment_size]
    gyro_seg = gyro_magnitudes[i:i+segment_size]
    #TODO: add acc magnitude and other features/data to segment
    segment_    = np.hstack([segment,seg_accmag, gyro_seg])

    features.append(segment_[:,:])
    label = Counter(raw_data_labels[i:i+segment_size][:, 0].tolist()).most_common(1)[0][0]
    labels.append(label)

  features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.33,
                                                    random_state=42)


  features_np_train = np.einsum('ijk->kij',np.dstack(features_train))
  labels_np_train   = np.array(labels_train)

  features_np_test = np.einsum('ijk->kij',np.dstack(features_test))
  labels_np_test   = np.array(labels_test)

  return features_np_train, labels_np_train,features_np_test, labels_np_test

def return_lstm_model(num_input_feat,num_classes):
    '''
    Returns a lstm model architecture
    Parameters
    ----------
    num_input_feat : integer
        Number of input features in the model
    num_classes : integer
        Number of output labels to be predicted
    Returns
    -------
    model : tf model
    '''
    _input   = tf.keras.layers.Input(shape=(None, num_input_feat))
    lstm1    = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128))(_input)
    dropout1 = tf.keras.layers.Dropout(rate=0.4)(lstm1)
    fc1      = tf.keras.layers.Dense(units=128,activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(rate=0.4)(fc1)
    fc2      = tf.keras.layers.Dense(units=64,activation='relu')(dropout2)
    out      = tf.keras.layers.Dense(units=num_classes,activation='softmax')(fc2)
    model    = tf.keras.Model(outputs=out, inputs=_input)
    return model

if __name__ == "__main__":


  data_path = 'data/'
  dirs = os.listdir(data_path)
  # print(dirs)

  X_train_lstm = []
  Y_train_lstm = []
  X_test_lstm = []
  Y_test_lstm = []

  X_lstm = []
  Y_lstm = []


  for dir in dirs:
      if dir == '.DS_Store':
          continue
      raw_data_features, raw_data_labels, timestamps = compute_raw_data(data_path + dir)
      print(raw_data_labels)
      features_lstm_train, labels_lstm_train, features_lstm_test, labels_lstm_test = feature_extraction_lstm_raw(raw_data_features,
                                                                                 raw_data_labels,
                                                                                 timestamps)

      X_train_lstm.append(features_lstm_train)
      Y_train_lstm.append(labels_lstm_train )
      X_test_lstm.append(features_lstm_test)
      Y_test_lstm.append(labels_lstm_test )
      
      print(features_lstm_test.shape)
     

X_test_lstm_np  = np.concatenate(X_test_lstm,axis=0)
Y_test_lstm_np  = np.concatenate(Y_test_lstm).squeeze()
X_train_lstm_np = np.concatenate(X_train_lstm,axis=0)
Y_train_lstm_np = np.concatenate(Y_train_lstm).squeeze()

#normalize features
scaler_fit        = StandardScaler().fit(X_train_lstm_np.reshape(-1,X_train_lstm_np.shape[-1]))
X_train_lstm_np_  = scaler_fit.transform(X_train_lstm_np.reshape(-1,X_train_lstm_np.shape[-1])).reshape(X_train_lstm_np.shape)
X_test_lstm_np_   = scaler_fit.transform(X_test_lstm_np.reshape(-1,X_test_lstm_np.shape[-1])).reshape(X_test_lstm_np.shape)

 
lstm_model = return_lstm_model(X_train_lstm_np_.shape[-1],
                           len(np.unique(Y_train_lstm_np)))
#set learning rate for the model 
lr         = 0.0001

adam_opt   = tf.keras.optimizers.Adam(lr=lr) 

lstm_model.compile(loss='categorical_crossentropy', optimizer=adam_opt,metrics=['accuracy'])

Y_train_lstm_np_ohe = ohe.fit_transform(Y_train_lstm_np[:,None]).toarray()
#model training

class_weight = dict(zip(np.arange(Y_train_lstm_np_ohe.shape[1]),
                    1/Y_train_lstm_np_ohe.sum(axis=0)))
print(class_weight)
lstm_model.fit(X_train_lstm_np_,Y_train_lstm_np_ohe,epochs=10,batch_size=64,
           class_weight=class_weight)

#predict the labels on the test set
pred_label    = lstm_model.predict(X_test_lstm_np_)
#inverse transform from one hot encoding to ordinal/categorical encoding
pred_label_np = ohe.inverse_transform(pred_label)
#evaluate accuracy scores
lstm_accu     = accuracy_score(Y_test_lstm_np,pred_label_np)
print('LSTM accu is: {0}'.format(lstm_accu))

target_names = pd.Series(activity_indices).to_frame().sort_values(by=0).index.values
generalizedmodel_classification_report= classification_report(Y_test_lstm_np,
                                                       pred_label_np,
                                                       target_names=target_names)
print(generalizedmodel_classification_report)
#save the model for loading in to flutter app
# Convert the model.
import pdb
#pdb.set_trace()
converter    = tf.lite.TFLiteConverter.from_keras_model(lstm_model)

tflite_model = converter.convert()


# Save the model.
with open('gesturerecog_model.tflite', 'wb') as f:
  f.write(tflite_model)
