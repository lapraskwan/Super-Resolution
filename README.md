# Single Image Super Resolution
A Tensorflow 2.0 implementation of several single image super resolution networks.

## Implemented Networks
1. [Image Super-Resolution Using Deep Convolutional Networks (SRCNN)](https://arxiv.org/pdf/1501.00092.pdf) (2014)
2. [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (ESPCN)](https://arxiv.org/pdf/1609.05158.pdf) (2016)
3. [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution (LapSRN)](https://arxiv.org/pdf/1704.03915.pdf) (2017)
4. [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)](https://arxiv.org/pdf/1609.04802.pdf) (2017)
5. [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/pdf/1809.00219.pdf) (2018)

## Results
For CNN methods the upscaled images are clear, but there are some artefacts for the GANs methods, which make the images look less realistic. In the future when time allows I will try to solve this problem.

## Versions
Tensorflow 2.7.0  
Python 3.9.7  
Numpy 1.19.5