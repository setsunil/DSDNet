# DSDNet: Discriminative Shrinkage Deep Network

The official pytorch implementation of the papers  
**[Efficient Non-Blind Image Deblurring with Discriminative Shrinkage Deep Networks (TCSVT2025)](https://ieeexplore.ieee.org/document/10937503)**  
**[Learning Discriminative Shrinkage Deep Networks for Image Deconvolution (ECCV2022)](https://arxiv.org/abs/2111.13876)**

#### [**Pin-Hung Kuo**](https://setsunil.github.io/), [Jinshan Pan](https://jspan.github.io/), [Shao-Yi Chien](https://www.ee.ntu.edu.tw/profile1.php?teacher_id=943013&p=3), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)

### Environments
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [NAFNet](https://github.com/megvii-research/NAFNet)
We tested our models in the following environments; higher versions may also be compatible.
```
python 3.8.20
pytorch 1.13.1
cuda 11.7
Anaconda
```
### Installation
Please install Anaconda first, then execute the following command:
```
git clone https://github.com/setsunil/DSDNet.git
cd DSDNet
source script.sh
```

### Usage
* Training
```
python -m torch.distributed.run --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/WEDBSD.yml --launcher pytorch
```

* Testing
```
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSDplus_p.yml --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSDplus_e.yml --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSD_full.yml --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSD_light.yml --launcher pytorch
```

### Datasets
[Google Drive](https://drive.google.com/file/d/1qNOTgEvm6Ag06YEc9b3iX2Zc9ZuIyTU1/view?usp=drive_link)

### Pretrained Models
[Google Drive](https://drive.google.com/file/d/1cZZ8Y6_z76_GSsBJAw50QFWZ8Clx8uou/view?usp=drive_link)

### Citations
If DSDNet helps your research or work, please consider citing us.

```
@inproceedings{kuo2022learning,
  title={Learning discriminative shrinkage deep networks for image deconvolution},
  author={Kuo, Pin-Hung and Pan, Jinshan and Chien, Shao-Yi and Yang, Ming-Hsuan},
  booktitle={European Conference on Computer Vision},
  pages={217--234},
  year={2022},
  organization={Springer}
}
```
```
@article{kuo2025efficient,
  title={Efficient Non-Blind Image Deblurring with Discriminative Shrinkage Deep Networks},
  author={Kuo, Pin-Hung and Pan, Jinshan and Chien, Shao-Yi and Yang, Ming-Hsuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

### Contact

If you have any questions, please contact setsunil@media.ee.ntu.edu.tw

---
<!--
<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=megvii-research/NAFNet)

</details>

-->