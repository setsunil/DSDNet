# Discriminative Shrinkage Deep Network



[**Pin-Hung Kuo**](https://setsunil.github.io/), [Jinshan Pan](https://jspan.github.io/), [Shao-Yi Chien](https://www.ee.ntu.edu.tw/profile1.php?teacher_id=943013&p=3), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)  
_[Media IC & System Lab](http://media.ee.ntu.edu.tw/), Taipei, Taiwan_

[[Paper](https://arxiv.org/pdf/2111.13876.pdf)]

Abstract
----------
Non-blind deconvolution is an ill-posed problem. Most existing methods usually formulate this problem into a maximum-a-posteriori framework and address it by designing kinds of regularization terms and data terms of the latent clear images. In this paper, we propose an effective non-blind deconvolution approach by learning discriminative shrinkage functions to implicitly model these terms. In contrast to most existing methods that use deep convolutional neural networks (CNNs) or radial basis functions to simply learn the regularization term, we formulate both the data term and regularization term and split the deconvolution model into data-related and regularization-related sub-problems according to the alternating direction method of multipliers. We explore the properties of the Maxout function and develop a deep CNN model with a Maxout layer to learn discriminative shrinkage functions to directly approximate the solutions of these two sub-problems. Moreover, given the fast Fourier transform based image restoration usually leads to ringing artifacts while conjugate gradient-based image restoration is time-consuming, we develop the conjugate gradient network to restore the latent clear images effectively and efficiently. Experimental results show that the proposed method performs favorably against the state-of-the-art ones in terms of efficiency and accuracy.


Data
----------
[Google Drive](https://drive.google.com/file/d/1xfDQ0OUmw8T5kralu-AhQ-eqPNZquUhc/view?usp=sharing)


Models
----------
[Google Drive](https://drive.google.com/file/d/1FHynxgJSXtTCQQVVSMcafwesNekSRrcG/view?usp=sharing)


Dependencies and Installation
----------
-Python >= 3.8
-[PyTorch >= 1.1](https://pytorch.org)
```
pip install -r requirement.txt
```

Reproduce the results in the paper
----------
```
python src/test.py -opt src/options/test/full_table2.yml
```
```
python src/test.py -opt src/options/test/light_table2.yml
```
```
python src/test.py -opt src/options/test/full_table3.yml
```


Citation
----------
```BibTex
 @article{kuo2021dsdnet,
  author    = {Kuo, Pin-Hung and Pan, Jinshan and Chien, Shao-Yi and Yang, Ming-Hsuan},
  title     = {Learning Discriminative Shrinkage Deep Networks for Image Deconvolution},
  url       = {https://arxiv.org/abs/2111.13876},
  eprinttype = {arXiv},
  eprint    = {2111.13876},
}
```


## Note
- This code is built upon the implementation from [MMSR](https://github.com/andreas128/mmsr).
