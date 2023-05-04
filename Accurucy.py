import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import dlib
import face_recognition
from PIL import Image, ImageDraw
from PIL import ImageFont

# Load known face encodings and names
image_maria = face_recognition.load_image_file("photo/maria.jpg")
maria_encoding_face = face_recognition.face_encodings(image_maria)[0]

image_maria1 = face_recognition.load_image_file("photo/maria1.jpg")

maria_encoding_face1 = face_recognition.face_encodings(image_maria1)[0]
# messi  

image_messi = face_recognition.load_image_file("photo/messi.jpg")
image_messi_face = face_recognition.face_encodings(image_messi)[0]

image_messi1 = face_recognition.load_image_file("photo/messi1.jpg")

image_messi1_encoding_face1 = face_recognition.face_encodings(image_messi1)[0]  

# moutouali 

image_mohssine1 = face_recognition.load_image_file("photo/mohssine1.jpg")
image_mohssine1_face = face_recognition.face_encodings(image_mohssine1)[0]

image_mohssine3 = face_recognition.load_image_file("photo/mohssine3.jpg")

image_mohssine3_encoding_face1 = face_recognition.face_encodings(image_mohssine3)[0]  

#boufal
image_soufiane = face_recognition.load_image_file("photo/soufiane.jpg")
image_soufiane_face = face_recognition.face_encodings(image_soufiane)[0]

image_soufiane1 = face_recognition.load_image_file("photo/soufiane1.jpg")

image_soufiane1_encoding_face1 = face_recognition.face_encodings(image_soufiane1)[0]  
#youssef
image_youssef = face_recognition.load_image_file("photo/youssef.jpg")
image_youssef_face = face_recognition.face_encodings(image_youssef)[0]

image_youssef2 = face_recognition.load_image_file("photo/youssef2.jpg")

image_youssef2_encoding_face1 = face_recognition.face_encodings(image_youssef2)[0]  







known_face_encoding = [maria_encoding_face ,maria_encoding_face1 , image_messi_face ,image_messi1_encoding_face1, image_mohssine1_face, image_mohssine3_encoding_face1,image_soufiane_face,image_soufiane1_encoding_face1,image_youssef_face,image_youssef2_encoding_face1]
known_face_name = ["maria", "maria", "messi", "messi", "mouhssine", "mouhssine","soufiane","soufiane","youssef","youssef"]

# Load unknown face encodings and names
image_1 = face_recognition.load_image_file("uknouwn/1.jpg")
image_1_face = face_recognition.face_encodings(image_1)[0]

image_5 = face_recognition.load_image_file("uknouwn/2.jpg")
image_5_face = face_recognition.face_encodings(image_5)[0]

image_6 = face_recognition.load_image_file("uknouwn/3.jpg")
image_6_face = face_recognition.face_encodings(image_6)[0]

image_7 = face_recognition.load_image_file("uknouwn/4.jpg")
image_7_face = face_recognition.face_encodings(image_7)[0]

image_9 = face_recognition.load_image_file("uknouwn/5.jpg")
image_9_face = face_recognition.face_encodings(image_9)[0]

image_10 = face_recognition.load_image_file("uknouwn/6.jpg")
image_10_face = face_recognition.face_encodings(image_10)[0]



uknown_face_encoding = [image_1_face, image_5_face, image_6_face, image_7_face, image_9_face, image_10_face]
uknown_face_name = ["youssef", "messi", "soufiane", "mouhssine", "maria", "shakira"]

# Convert face encodings to numpy arrays
known_face_encoding_array = [i.flatten() for i in known_face_encoding]
uknown_face_encoding_array = [i.flatten() for i in uknown_face_encoding]

# Split data into training and testing sets
x_train = np.array(known_face_encoding_array)
y_train = known_face_name
x_test = np.array(uknown_face_encoding_array)
y_test = uknown_face_name

# Initialize k-NN classifier with k=1
knn = KNeighborsClassifier(n_neighbors=1)

# Fit k-NN classifier on training data
knn.fit(x_train, y_train)

# Predict labels for test data
y_pred = knn.predict(x_test)

# Calculate accuracy of k-NN classifier on test data
accuracy = np.mean(y_pred == y_test) * 100
print("Accuracy: {:.2f}%".format(accuracy))