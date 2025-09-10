# Face-Recognition :boy: :movie_camera: 
Face Recognition using KNN algorithm and open cv for python.This is a implementation of knn classifier.
## Breakdown of the code for knn classifier
    1. Importing libraries
    2. Create some data for classification
    3. Write the kNN workflow
    4. Finally, run knn on the data and observe results

## Face Recognition Pipeline
    Data Collection (record_faces.py)
    Face Detection (Haar Cascade Classifiers)
    Feature Extraction (Image Processing)
    Machine Learning (KNN Algorithm)
    Real-time Recognition (face_recog.py)

## Project Structure
    record_faces.py          # Data collection and training module
    face_recog.py           # Main recognition engine
    haarcascade_frontalface_default.xml  # Primary face detector
    haarcascade_frontalface_alt.xml      # Alternative face detector
    face_01.npy             # Sample face data 1
    face_02.npy             # Sample face data 2
## Install VSCode extension (for .npy file view)
    vscode-pydata-viewer
## Dependencies
    Python 2.7, Numpy and OpenCv
## How it works! :wink:  
* Run record_faces.py on the command line.The script will open a camera window.Stand in front of the camera until recording of the face is completed.
* The default file where the features are stored is face_01.npy. You can change the file name if you want to store information of many persons.It stores data in a numpy matrix.
* Open the face_recognition.py file and edit your name in the dictionary value corresponding to the number in which your face was stored i.e. for face_01,add your name to '0' value in the names dictionary.
* Run the face_recognition.py file!


# Clone repository
    git clone https://github.com/prashant0598/Face-Recognition.git
    cd Face-Recognition
# Install uv venv
    uv pip install ruff
    source .venv/bin/activate
    deactivate

# Optional: if not find lib's
    uv pip install Numpy
    uv pip install OpenCv

# Run data collection
    python record_faces.py

# Run recognition
    python face_recog.py
## Accuracy :tada:
   * 98.4 (using knn) because of small dataset. 
   * Taking distance from webcam and quality of light into consideration it would give 90+ accuracy.
