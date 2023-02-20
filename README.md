#BatFinder Smart Video BioFilter (BFSVBF) and Multi-class BatFinder Smart Video BioFilter
Keras(TF backend) developed biological object detector used to detect biological objects flying within the vicinity of Wind Turbines using Thermal Cameras as detection

## Requirements
See requirments text in the repository
- opencv-contrib-python==3.4.5.20
- Keras == 2.3.1
- Pillow == 6.
- Numpy = 1.17.2
- Matplotlib == 3.1.1
- Tensorflow == 2.0.0

## Description

There are two object classifier machine learning models, Binary and multi-classification.  

Binary object classifier labeled BatFinder_Smart_Video_BioFilter.h5 distinguishes between biological objects and non-biological objects.  The main goal of this object classifier is to ignore the turbine blades while detecting biological object flying withing the rotor swept area of the turbine.  Non-biological objects have a probability of 0 and biological objects have a probability of 1.

Multi-classifier labeled Multiclass_BatFinder_Smart_Video_BioFilter.h5 distinguishes between bats, birds, insects and non-biological.

## Copyright
See [License](License) for details



