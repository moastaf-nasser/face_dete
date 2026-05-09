import cv2
import numpy as np
import face_recognition
import os

# Configuration and Path setup
path = "images"
images = []
class_names = []
mylist = os.listdir(path)

# Load training images and extract class names from file names
for cl in mylist:
    current_image = cv2.imread(f'{path}/{cl}')
    if current_image is not None:
        images.append(current_image)
        class_names.append(os.path.splitext(cl)[0])

def find_encodings(images):
    """
    Function to generate 128-dimensional face encodings for the known database.
    """
    encode_list = []
    for img in images:
        # Convert BGR (OpenCV default) to RGB (Face_recognition requirement)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encode_list.append(encodes[0])
    return encode_list

# Initialize the encoding process for known faces
encode_list_known = find_encodings(images)
print('Database Encoding Successful.')

# Initialize webcam stream
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Optimization: Resize frame to 1/4 to increase processing FPS
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # Detect faces and generate encodings for the current frame
    faces_cur_frame = face_recognition.face_locations(img_small)
    encodes_cur_frame = face_recognition.face_encodings(img_small, faces_cur_frame)

    # Compare detected faces against the known database
    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        
        # Identify the best match based on the lowest Euclidean distance
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = class_names[match_index].upper()
            
            # Rescale coordinates back to the original image size (4x)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            # Visual Feedback: Draw bounding box and name label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Display the processed output
    cv2.imshow('Biometric Authentication System', img)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Resource Cleanup
cap.release()
cv2.destroyAllWindows()