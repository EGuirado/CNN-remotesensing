# CNN-remotesensing

Using CNN models (ResNet-152 and Inception v3) to detect shrubs in satellite imagery (Google Earth)

Due to space limitation in Github, the dataset and codes are provided via email request to this email: e.guirado@ual.es and siham@ugr.es

## Dataset
The training dataset must show this structure:
```
Train/
      class1/
             class1_image0.jpg
             class1_image1.jpg
             .
             .
             .
      classN/
             classN_image0.jpg
             classN_image1.jpg
             .
             .
             .
Validation/
          class1/
                class1_image0.jpg
                class1_image1.jpg
                .
                .
                .
          classN/
                classN_image0.jpg
                classN_image1.jpg
                .
                .
                .
```  

## Code
Retrain [Inception v3](retrain_inceptionv3.py) model with new repository data.
Required:
- Tensorflow v.0.10

```bash
python retrain_inceptionv3.py --bottleneck_dir=tests/bottlenecks-100stZ_S-105-299 --how_many_training_steps 100 --model_dir=inception --output_graph=tests/retrained_graph-100stZ_S-105-299.pb --output_labels=tests/retrained_labels-100stZ_S-105-299.txt --image_dir=datasets/
```

Sliding window Heatmap
Required:
- Pillow
- matplotlib
- numpy

```bash
```

Preprocessing data and clasify
Required:
- Opencv2
- Python2.7
```bash
```
.
.
.

## acknowledgements
The study of shrub detection with Google Earth imagery was based in Google developers for the [Inception](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2) model and adaptation of [ResNet-152](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6) model used by [Keras]( https://keras.io) and [Tensorflow](https://www.tensorflow.org/).
