import cv2
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, StratifiedKFold
from sklearn.metrics import  accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# Use four pre-trained classifiers for face detection
face_detector_1 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
face_detector_2 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
face_detector_3 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt2.xml')
face_detector_4 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt_tree.xml')


emotion_labels = {'Neutral': 0,
                  'Anger': 1,
                  'Surprise': 2,
                  'Sadness': 3,
                  'Happy': 4}

# add your photos to the folder and set your netid
NetID = 'ma98'


def feature_extraction(img, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
  """ The function does the following tasks to extract emotion-related features:
      (1) Face detection (2) Cropping the face in the image (3) Resizing the image and (4) Extracting HOG vector.

    Args:
      img: The raw image.
      orientations: The number of bins for different orientations.
      pixels_per_cell: The size of each cell.
      cells_per_block: The size of the block for block normalization.

    Returns:
      features: A HOG vector is returned if face is detected. Otherwise 'None' value is returned.
  """

  # If the image is a color image, convert it into gray-scale image
  if img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


  face_detection_1 = face_detector_1.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
  face_detection_2 = face_detector_2.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
  face_detection_3 = face_detector_3.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
  face_detection_4 = face_detector_4.detectMultiScale(
    img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)


  # Go over the results of face detection. Stop at the first detected face,
  face_features = None
  if len(face_detection_1) == 1:
    face_features = face_detection_1
  elif len(face_detection_2) == 1:
    face_features = face_detection_2
  elif len(face_detection_3) == 1:
    face_features = face_detection_3
  elif len(face_detection_4) == 1:
    face_features = face_detection_4
  else:
    print("No face detected!")
    # cv2.imshow('No face detected', img)
    # cv2.waitKey(0)


  if face_features is not None:
      global count
      for x, y, w, h in face_features:
          # Get the coordinates and the size of the rectangle containing face
          img = img[y:y+h, x:x+w]
        
          # Resize all the face images so that all the images have the same size
          img = cv2.resize(img, (350, 350))
          # Uncomment the following two lines to visualize the cropped face image
          # cv2.imshow("Cropped Face", img)
          # cv2.waitKey(0)
        
          # Extract HOG descriptor
          features, hog_img = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)
          # Uncomment the following tow lines to visualize HOG
          #cv2.imshow('hog', hog_img)
          #cv2.waitKey(0)
          count += 1
          #print("Loading: {:d}%".format(int(count / 50 * 100)))
          return features.reshape(1, -1)

  else:
      return None




if __name__ == "__main__":


  "***Feature Extraction***"

  # Dictionary whose <key, value> is <user, (features, labels)>
  dataset = dict()

  path = './images'

  # Get all the folder of individuad subject
  cellsize = [4, 8, 16, 32, 64]
  for cell_s in cellsize:
      for subject in os.listdir(path):
        if subject[0] == '.':
          continue
        print(subject)
        count = 0
        emotion_dirs = os.listdir(path + '/%s' %subject)
        feature_matrix = None
        labels = None
    
        for emotion_dir in emotion_dirs:
          if emotion_dir[0] == '.':
            continue
          # Get the index associated with the emotion
          emotion_label = emotion_labels[emotion_dir]
    
          for f in os.listdir(path + '/%s/%s' %(subject, emotion_dir)):
            img = cv2.imread(path + '/%s/%s/' %(subject, emotion_dir) + f)
            # Uncomment the following two lines to visualize the raw images
            # cv2.imshow("raw img", img)
            # cv2.waitKey(0)
    
            # Extract HOG features
            
            features = feature_extraction(img, orientations=64, pixels_per_cell=(cell_s, cell_s), cells_per_block=(1, 1))
    
            if features is not None:
              feature_matrix = features if feature_matrix is None else np.append(feature_matrix, features, axis=0)
              labels = np.array([emotion_label]) if labels is None else np.append(labels, np.array([emotion_label]), axis=0)
    
    
        dataset[subject] = (feature_matrix, labels)
    
    
      "***Person-dependent Model***"
      X, y = dataset[NetID]
    
      # TODO: Use leave-one-subject-out cross validation to evaluate the generalized (person-independent) models.
      # You will need to train a model on data from different sets of people and predict the remaining person's emotion.
      X_train = []
      y_train = []
      for subject in os.listdir(path):
            if subject == 'ma98' or subject == '.DS_Store':
              X_test, y_test = dataset['ma98']
              continue
            print(subject)
            X, y = dataset[subject]
            X_train.append(X)
            y_train.append(y)
      X_tr = np.vstack(X_train)
      y_tr = np.hstack(y_train)
        
      clf = svm.SVC()
      clf.fit(X_tr, y_tr)
      y_hat = clf.predict(X_test)
        
        
      print('accuracy', accuracy_score(y_test, y_hat))
      print('precision', precision_score(y_test, y_hat, average='micro'))
      print('recall', precision_score(y_test, y_hat, average='macro'))
      confusion = confusion_matrix(y_test, y_hat)
      print('confusion', confusion)    

