import numpy as np
import cv2

# instantiate the camera object and haar cascade
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# declare the type of font to be used on output window
font = cv2.FONT_HERSHEY_SIMPLEX

# load the data from the numpy matrices and convert to linear vectors
f_01 = np.load('face_01.npy').reshape((20, 50*50*3))    # face detected
f_02 = np.load('face_02.npy').reshape((20, 50*50*3))    # face confusion with old data

print(f_01.shape, f_02.shape)

# create a look-up dictionary
names = {
    0: 'face detected',
    1: 'face confusion with old data',
}

# create a matrix to store the labels
labels = np.zeros((40, 1))
labels[:20, :] = 0.0    # first 20 for Prashant (0)
labels[20:40, :] = 1.0  # next 20 for Nil (1)

# combine all info into one data array
data = np.concatenate([f_01, f_02]) # (40, 7500)
print(data.shape, labels.shape)  # (40, 1)

# Normalize the data
data = data.astype(np.float32) / 255.0

# Improved KNN implementation with unknown detection
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_with_unknown(x, train, targets, k=5, unknown_threshold=0.6):
    """
    KNN that can detect unknown faces
    Returns: (predicted_label, confidence, is_unknown)
    """
    distances = []
    for i in range(train.shape[0]):
        dist = distance(x, train[i])
        distances.append((dist, targets[i]))
    
    # Sort by distance and get k nearest neighbors
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    # Calculate average distance to neighbors
    avg_distance = np.mean([dist for dist, _ in neighbors])
    
    # Count votes for each class
    votes = {}
    for dist, label in neighbors:
        votes[label] = votes.get(label, 0) + 1
    
    # Get the winning class and its vote count
    if votes:
        winner_label = max(votes.items(), key=lambda x: x[1])[0]
        winner_votes = votes[winner_label]
        confidence = winner_votes / k
        
        # Check if it's an unknown face based on distance threshold
        if avg_distance > unknown_threshold:
            return -1, confidence, True  # Unknown face
        else:
            return winner_label, confidence, False  # Known face
    else:
        return -1, 0.0, True  # Unknown face

# Calculate average distances within known faces for threshold setting
def calculate_reference_threshold(train_data, train_labels, k=3):
    """
    Calculate a good threshold for unknown face detection
    """
    intra_class_distances = []
    
    for i in range(train_data.shape[0]):
        # Find distances to other samples of same class
        same_class_indices = np.where(train_labels == train_labels[i])[0]
        same_class_indices = same_class_indices[same_class_indices != i]  # Exclude self
        
        if len(same_class_indices) > 0:
            distances = []
            for j in same_class_indices:
                dist = distance(train_data[i], train_data[j])
                distances.append(dist)
            
            # Take average distance to k nearest same-class neighbors
            distances.sort()
            intra_class_distances.extend(distances[:min(k, len(distances))])
    
    if intra_class_distances:
        avg_intra_distance = np.mean(intra_class_distances)
        std_intra_distance = np.std(intra_class_distances)
        # Set threshold as average + 2 standard deviations
        threshold = avg_intra_distance + 2 * std_intra_distance
        return max(threshold, 0.7)  # Minimum threshold of 0.7
    else:
        return 0.8  # Default threshold

# Calculate optimal threshold
labels_flat = labels.flatten()
UNKNOWN_THRESHOLD = calculate_reference_threshold(data, labels_flat)
print(f"Unknown detection threshold: {UNKNOWN_THRESHOLD:.3f}")

while True:
    # get each frame
    ret, frame = cam.read()

    if ret == True:
        # convert to grayscale and get faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)

        # for each face
        for (x, y, w, h) in faces:
            face_component = frame[y:y+h, x:x+w, :]
            fc = cv2.resize(face_component, (50, 50))
            
            # Preprocess the test image
            test_image = fc.flatten().astype(np.float32) / 255.0
            
            # Use our improved KNN with unknown detection
            lab, confidence, is_unknown = knn_with_unknown(
                test_image, data, labels_flat, 
                k=5, unknown_threshold=UNKNOWN_THRESHOLD
            )
            
            # Display results
            if is_unknown or lab == -1:
                text = "Unknown Person"
                color = (0, 0, 255)  # Red for unknown
                status = "NOT in database"
            else:
                text = names[int(lab)]
                color = (0, 255, 0)  # Green for recognized
                status = "Recognized"
            
            # Display the information
            cv2.putText(frame, f"{text}", (x, y-30), font, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y-10), font, 0.5, color, 1)
            cv2.putText(frame, f"Status: {status}", (x, y+h+20), font, 0.5, color, 1)
            
            # draw a rectangle over the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        cv2.imshow('Face Recognition - Press ESC to exit', frame)

        if cv2.waitKey(1) == 27:
            break
    else:
        print('Error reading camera')

cam.release()
cv2.destroyAllWindows()