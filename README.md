# YOLO for Gender and Pakistani Celebrities

This project utilizes computer vision techniques to detect human faces, classify their gender, and recognize specific Pakistani celebrities. The project comprises two main components, structured within Jupyter Notebooks, emphasizing training and inference using YOLOv8 and ResNet-50.

## Overview

The primary objective of this project is to create an integrated pipeline capable of:
1. Detecting faces in an image using YOLOv8.
2. Formulating a bounding box and identifying the gender (Male/Female) of the detected person.
3. Classifying the identity of the person among several famous Pakistani celebrities.

## Project Structure

*   `YOLO_Train.ipynb`: This notebook involves training a custom YOLOv8 model for gender classification (Female/Male). It covers setting up the dataset, training configurations, and plotting the evaluation graphs using Ultralytics.
*   `CV_Part_2_main_Final_ (1).ipynb`: The primary inference and classification notebook. It loads the customized YOLOv8 model (for face detection and gender classification) alongside a fine-tuned TensorFlow/Keras application model (ResNet50) for targeted celebrity face recognition.

## Dependencies

You need the following main libraries to run the models:
*   `ultralytics`
*   `opencv-python` (cv2)
*   `torch` and `torchvision` (PyTorch)
*   `tensorflow` and `keras`
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`

To ensure a smooth setup, install the dependencies using pip:
```bash
pip install ultralytics opencv-python torch torchvision tensorflow numpy pandas matplotlib seaborn
```

## Classes Included

The facial recognition module can predict the following identities:
*   Iqra Aziz
*   Hamuyun Saeed
*   Atif Aslam
*   Fahad Mustafa
*   Fawad Khan
*   Hamza Ali Abbasi
*   Hania Amir
*   Qubra Khan
*   Maira Khan
*   Naseem Shah
*   Noman Ijaz
*   Neelam Muneer
*   Ramsha Khan
*   Sajal Ali
*   Shaheen Shah Afridi
*   (0: Unknown)

## Usage

1. Open `CV_Part_2_main_Final_ (1).ipynb`.
2. Ensure you have the trained weights correctly referenced (`/content/best (1).pt` for YOLO and `/content/best_model_f1.keras` for the ResNet classifier).
3. Update the path of the test image pointing in `detect_and_classify_faces('/path/to/your/image.jpg')`.
4. Run the notebook to get an output image (`annotated_output.jpg`) highlighting bounding boxes, confidence scores, gender, and celebrity names.
