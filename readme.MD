#  Fast Image Processing with Fully-Convolutional Networks
This is a Tensorflow implementation of Fast Image Processing with Fully-Convolutional Networks.

## Demo Video
https://www.youtube.com/watch?v=eQyfHgLx8Dc

## Setup

### Requirement
Required python libraries: Tensorflow (>=1.0) + Opencv + Numpy.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.

### Quick Start (Testing)
1. Clone this repository.
2. Run "CAN24_AN/demo.py". This will generate results on L0 smoothing in "CAN24_AN/L0_smoothing/MIT-Adobe_test_1080p_result".
3. To test a different model, change the variable "task" in "demo.py"

### Training
1. To train, change "is_training" to "True".
2. To set up a customized training procedure, change the file paths in "prepare_data()". See the commands in the code.

## Extensions
1. The single network for all operators is "combined.py" in the folder "Single_Network". Run it and its result is in "Single_Network/result_combined/video".
2. The parameterized network is "parameterized.py" in the folder "Parameterized_Network". Run it and its result is in "Parameterized/result_parameterized/video".

## Data
If you want to experiment on the data in our evaluation, please email to chenqifeng22@gmail.com.

## Citation
If you use our code for research, please cite our paper:

Qifeng Chen, Jia Xu, and Vladlen Koltun. Fast Image Processing with Fully-Convolutional Networks. In ICCV 2017.

### License
MIT License.



