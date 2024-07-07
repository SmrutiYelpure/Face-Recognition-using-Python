# Face Recognition Project

This project implements a face recognition system using the Labeled Faces in the Wild (LFW) dataset. It uses dlib for face detection and feature extraction, and a Support Vector Machine (SVM) for classification.

## Project Structure
<ul>
  <li><a href="#data">data/</a>
    <ul>
      <li><a href="#lfw-funneled">lfw-funneled.zip</a></li>
      <li><a href="#pairs">pairs.txt</a></li>
      <li><a href="#pairsDevTest">pairsDevTest.txt</a></li>
      <li><a href="#pairsDevTrain">pairsDevTrain.txt</a></li>
    </ul>
  </li>
  <li><a href="#src">src/</a>
    <ul>
      <li><a href="#utils">utils.py</a></li>
      <li><a href="#prepare_dataset">prepare_dataset.py</a></li>
      <li><a href="#train_model">train_model.py</a></li>
      <li><a href="#recognize_face">recognize_face.py</a></li>
    </ul>
  </li>
  <li><a href="#models">models/</a></li>
  <li><a href="#venv">venv/</a></li>
  <li><a href="#requirements">requirements.txt</a></li>
  <li><a href="#README.md">README.md</a></li>
  <li><a href="#shape_predictor_68_face_landmarks">shape_predictor_68_face_landmarks.dat</a></li>
  <li><a href="#dlib_face_recognition_resnet_model_v1">dlib_face_recognition_resnet_model_v1.dat</a></li>
</ul>

## Setup

1. Clone the repository:
git clone https://github.com/SmrutiYelpure/Face-Recognition-using-Python.git
cd face-recognition-project
Copy
2. Create a virtual environment:
python -m venv venv
Copy
3. Activate the virtual environment:
- On Windows: `venv\Scripts\activate`
- On macOS/Linux: `source venv/bin/activate`

4. Install the required packages:
pip install -r requirements.txt
Copy
5. Download the required dlib model files:
- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

Place these files in the project root directory.

6. Download the LFW dataset:
- [LFW-funneled](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)

Extract and place the `lfw-funneled.zip` file in the `data/` directory.

## Usage

1. Prepare the dataset:
python src/prepare_dataset.py
CopyThis script processes the LFW dataset and creates face embeddings.

2. Train the model:
python src/train_model.py
CopyThis script trains an SVM classifier on the prepared dataset.

3. Recognize faces:
python src/recognize_face.py <name1> <image_number1> <name2> <image_number2>
CopyFor example:
python src/recognize_face.py George_W_Bush 0001 George_W_Bush 0002
CopyThis script compares two face images and determines if they belong to the same person.

## File Descriptions

- `utils.py`: Contains utility functions for loading images, creating face embeddings, and handling the dataset.
- `prepare_dataset.py`: Processes the LFW dataset and creates face embeddings.
- `train_model.py`: Trains an SVM classifier on the prepared dataset.
- `recognize_face.py`: Implements face recognition using the trained model.

## Model Details

The face recognition system uses the following components:

1. Face Detection: dlib's frontal face detector
2. Facial Landmark Detection: dlib's 68-point shape predictor
3. Face Embedding: dlib's face recognition model (ResNet network)
4. Classification: Support Vector Machine (SVM) with RBF kernel

The model is trained on pairs of faces from the LFW dataset, learning to distinguish between pairs of the same person and pairs of different people.

## Performance

The model's performance is evaluated using accuracy, precision, recall, and F1-score. A ROC curve is also generated to visualize the trade-off between true positive rate and false positive rate.

## Future Improvements

- Implement data augmentation to increase the dataset size
- Experiment with different classifiers or ensemble methods
- Fine-tune the face recognition model for the specific dataset
- Implement cross-validation for more robust evaluation

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset
- [dlib](http://dlib.net/) library for face detection and recognition
