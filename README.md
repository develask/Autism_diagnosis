# Autism diagnosis with Neural Networks

On this project, we try to diagnose autism patients from from MRI's raw input data.

To do this job, we get data from Abide and Abide II:

* [ABIDE Data](http://fcon_1000.projects.nitrc.org/indi/abide/ABIDE_LEGEND_V1.02.pdf)
* [ABIDE II Phenotypic Data](http://fcon_1000.projects.nitrc.org/indi/abide/ABIDEII_Data_Legend.pdf)

*This is only a quick view of how Neural Network can help on these kind of problems, so we decided to take a small images (6mm) just to simplify the network, and not to spend long time on the training process the net.

All software and data is free, so every one can use it!

## Let's do it!

We will do it in Python3, so first we will need next librarys:

* [Nibabel](http://nipy.org/nibabel/manual.html#manua l): Read MRI images (*.nii.gz)
* [Numpy](http://www.numpy.org/): Work with images data
* [Keras](https://keras.io/): For Neural Networks (NN) implementation. We will use [TensorFlow](https://www.tensorflow.org/) as backend.
* [MatPlotLib](http://matplotlib.org/): Plot different data results.

### Explaining oue data structure

Next to this notebook we have a folder (named ABIDE) with all ABIDE data aviable, with all this data:

* Autism&#x268A;diagnosis.ipynb
* ABIDE/
    * Phenotypic&#x268A;V1&#x268A;0&#x268A;b.csv
    * ABIDEII&#x268A;CompositeP&#x268A;henotypic.csv
    * weighted&#x268A;degree/
        * (SITE&#x268A;ID)&#x268A;(SUB&#x268A;ID).nii.gz
    * weighted&#x268A;degree&#x268A;II/
        * (SITE&#x268A;ID)&#x268A;(SUB&#x268A;ID).nii.gz
* 6mm/
    * avg152T1&#x268A;(mask&#x268A;type)&#x268A;bin.nii.gz
    
#### Reading FMR images names for each location

We will read both images directories and save each image name in a `<names>` variable.