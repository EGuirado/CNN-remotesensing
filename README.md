# CNN-remotesensing

Using CNN models (ResNet and Inception v3) to detect shrubs in satellite imagery (Google Earth)

## Dataset
The training dataset must show this structure:
'''
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
'''  

## Code



The study of shrub detection with Google Earth images was based on the projects [by Google developers](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2) for the Inception model and adaptation of [ResNet-152](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6) model used by [Keras]( https://keras.io) and [Tensorflow](https://www.tensorflow.org/).
