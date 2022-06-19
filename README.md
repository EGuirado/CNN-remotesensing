# CNN-remotesensing

The study of shrub detection with Google Earth imagery was based on Inception model and  ResNet-152 model using Keras(as front-end) and Tensorflow(as back-end).

Due to space limitation in Github, the dataset and codes are provided via Google Drive: https://drive.google.com/drive/u/0/folders/1IfyivDERnNj-NW6q6CCam47zca-n0p6e

## Short description:
############ INCEPTION v3 ############
## Directory structure
Inceptionv3
1_Dataset
classes (database to train with class images in folders. e.g. “Z” for vegetation and “S”

for soil)

image (image to classification test. e.g. “ZoneTest.jpg”)
2_Train
inception (Model pretrained weights inception v3 )
Train_models (Model retrained weights inception v3 + new classes chips “Z” and “S” )
3_Label (3 options)
individual (show probabilities in one test image like images in 1_Dataset/classes/Z/

“label_image_new.py” Tensorflow 1.0 or greater)

preprocessing (show probabilities in bounding box for extent test image like

“ZoneTest.jpg”. Requires OpenCV2)

heatmap (show heatmap of extent test image like “ZoneTest.jpg”)

# Requisites and libs for inception v3 model
Pillow
Tensorflow >= 1.0
Opencv + numpy
python 2.7
## How to retrain model inception v3 with augmentation data
python 2_Train/retrain_au-scale.py --bottleneck_dir=2_Train/Train_models/bottlenecks --
how_many_training_steps 1000 --model_dir=inception --
output_graph=2_Train/Train_models/retrained_graph.pb --
output_labels=Train/Train_models/retrained_labels.txt --image_dir=1_Dataset/classes
## How to label test individual image
python 3_Label/individual/label_image_new.py 1_Da taset/test/Z/Z1.jpg
## How to label with preprocessing an image
python 3_Label/preprocessing/preprocessing.py
## How to label with Heatmap (2 steps)
1-create_csv_from_slidingwindow.py
2-heatmap_from_csv.py

########### RESNET 152 ###########
## Directory structure

ResNet152
1_Dataset
Train
Z
S
Test
Z
S

# Requisites and libs for ResNet 152 model
Keras >= 2 https://keras.io/
Tensorflow >= 1.0 https://www.tensorflow.org/
Opencv + numpy
python 2.7
#ResNet model train and validate test images.
python resnet_152.py --image_dir=1_Dataset/ --imgs_rows=224 --imgs_cols=224 --
batch_size=8 --epochs=10


Corresponding e.guirado@ual.es and siham@ugr.es


## Acknowledgements
The study of shrub detection with Google Earth imagery was based in Google developers for the [Inception](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2) model and adaptation of [ResNet-152](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6) model used by [Keras]( https://keras.io) and [Tensorflow](https://www.tensorflow.org/).
