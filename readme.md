## DSDNet: Discriminative Shrinkage Deep Network

The official pytorch implementation of the papers  
**[Efficient Non-Blind Image Deblurring with Discriminative Shrinkage Deep Networks (TCSVT2025)](https://ieeexplore.ieee.org/document/10937503)**
**[Learning Discriminative Shrinkage Deep Networks for Image Deconvolution (ECCV2022)](https://arxiv.org/abs/2111.13876)**

<!--Pin-Hung Kuo\*, Jinshan Pan, Shao-Yi Chien, Ming-Hsuan Yang-->
#### [**Pin-Hung Kuo**](https://setsunil.github.io/), [Jinshan Pan](https://jspan.github.io/), [Shao-Yi Chien](https://www.ee.ntu.edu.tw/profile1.php?teacher_id=943013&p=3), and [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)
<!--
>Although there have been significant advances in the field of image restoration recently, the system complexity of the state-of-the-art (SOTA) methods is increasing as well, which may hinder the convenient analysis and comparison of methods.
>In this paper, we propose a simple baseline that exceeds the SOTA methods and is computationally efficient.
>To further simplify the baseline, we reveal that the nonlinear activation functions, e.g. Sigmoid, ReLU, GELU, Softmax, etc. are **not necessary**: they could be replaced by multiplication or removed. Thus, we derive a Nonlinear Activation Free Network, namely NAFNet, from the baseline. SOTA results are achieved on various challenging benchmarks, e.g. 33.69 dB PSNR on GoPro (for image deblurring), exceeding the previous SOTA 0.38 dB with only 8.4% of its computational costs; 40.30 dB PSNR on SIDD (for image denoising), exceeding the previous SOTA 0.28 dB with less than half of its computational costs.

| <img src="./figures/denoise.gif"  height=224 width=224 alt="NAFNet For Image Denoise"> | <img src="./figures/deblur.gif" width=400 height=224 alt="NAFNet For Image Deblur"> | <img src="./figures/StereoSR.gif" height=224 width=326 alt="NAFSSR For Stereo Image Super Resolution"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           Denoise                            |                            Deblur                            |                           StereoSR([NAFSSR](https://github.com/megvii-research/NAFNet/blob/main/docs/StereoSR.md))                           |

![PSNR_vs_MACs](./figures/PSNR_vs_MACs.jpg)

### News
**2022.08.02** The Baseline, including the pretrained models and train/test configs, are available now.

**2022.07.03** Related work, [Improving Image Restoration by Revisiting Global Information Aggregation](https://arxiv.org/abs/2112.04491) (TLC, a.k.a TLSC in our paper) is accepted by **ECCV2022** :tada: . Code is available at https://github.com/megvii-research/TLC.

**2022.07.03** Our [paper](https://arxiv.org/abs/2204.04676) is accepted by **ECCV2022** :tada:

**2022.06.19** [NAFSSR](https://arxiv.org/abs/2204.08714) (as a challenge winner) is selected for an ORAL presentation at CVPR 2022, NTIRE workshop  :tada: [Presentation video](https://drive.google.com/file/d/16w33zrb3UI0ZIhvvdTvGB2MP01j0zJve/view), [slides](https://data.vision.ee.ethz.ch/cvl/ntire22/slides/Chu_NAFSSR_slides.pdf) and [poster](https://data.vision.ee.ethz.ch/cvl/ntire22/posters/Chu_NAFSSR_poster.pdf) are available now.

**2022.04.15** NAFNet based Stereo Image Super-Resolution solution ([NAFSSR](https://arxiv.org/abs/2204.08714)) won the **1st place** on the NTIRE 2022 Stereo Image Super-resolution Challenge! Training/Evaluation instructions see [here](https://github.com/megvii-research/NAFNet/blob/main/docs/StereoSR.md).
-->
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
```
git clone https://github.com/setsunil/DSDNet.git
cd DSDNet
source script.sh
```

### Usage
* Train
```
python -m torch.distributed.run --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/WEDBSD.yml --launcher pytorch
```

* Test
```
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSDplus_p.yml --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSDplus_e.yml --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSD_full.yml --launcher pytorch
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/test.py -opt options/test/DSD_light.yml --launcher pytorch
```
<!--
### Quick Start
* Image Denoise Colab Demo: [<a href="https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing)
* Image Deblur Colab Demo: [<a href="https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing)
* Stereo Image Super-Resolution Colab Demo: [<a href="https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing)
* Single Image Inference Demo:
    * Image Denoise:
    ```
    python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png
  ```
    * Image Deblur:
    ```
    python basicsr/demo.py -opt options/test/REDS/NAFNet-width64.yml --input_path ./demo/blurry.jpg --output_path ./demo/deblur_img.png
    ```
    * ```--input_path```: the path of the degraded image
    * ```--output_path```: the path to save the predicted image
    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded.
    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for single image restoration[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFNet)
* Stereo Image Inference Demo:
    * Stereo Image Super-resolution:
    ```
    python basicsr/demo_ssr.py -opt options/test/NAFSSR/NAFSSR-L_4x.yml \
    --input_l_path ./demo/lr_img_l.png --input_r_path ./demo/lr_img_r.png \
    --output_l_path ./demo/sr_img_l.png --output_r_path ./demo/sr_img_r.png
    ```
    * ```--input_l_path```: the path of the degraded left image
    * ```--input_r_path```: the path of the degraded right image
    * ```--output_l_path```: the path to save the predicted left image
    * ```--output_r_path```: the path to save the predicted right image
    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded.
    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for stereo image super-resolution[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFSSR)
* Try the web demo with all three tasks here: [![Replicate](https://replicate.com/megvii-research/nafnet/badge)](https://replicate.com/megvii-research/nafnet)

### Results and Pre-trained Models

| name | Dataset|PSNR|SSIM| pretrained models | configs |
|:----|:----|:----|:----|:----|-----|
|NAFNet-GoPro-width32|GoPro|32.8705|0.9606|[gdrive](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1AbgG0yoROHmrRQN7dgzDvQ?pwd=so6v)|[train](./options/train/GoPro/NAFNet-width32.yml) \| [test](./options/test/GoPro/NAFNet-width32.yml)|
|NAFNet-GoPro-width64|GoPro|33.7103|0.9668|[gdrive](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1g-E1x6En-PbYXm94JfI1vg?pwd=wnwh)|[train](./options/train/GoPro/NAFNet-width64.yml) \| [test](./options/test/GoPro/NAFNet-width64.yml)|
|NAFNet-SIDD-width32|SIDD|39.9672|0.9599|[gdrive](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1Xses38SWl-7wuyuhaGNhaw?pwd=um97)|[train](./options/train/SIDD/NAFNet-width32.yml) \| [test](./options/test/SIDD/NAFNet-width32.yml)|
|NAFNet-SIDD-width64|SIDD|40.3045|0.9614|[gdrive](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/198kYyVSrY_xZF0jGv9U0sQ?pwd=dton)|[train](./options/train/SIDD/NAFNet-width64.yml) \| [test](./options/test/SIDD/NAFNet-width64.yml)|
|NAFNet-REDS-width64|REDS|29.0903|0.8671|[gdrive](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1vg89ccbpIxg3mK9IONBfGg?pwd=9fas)|[train](./options/train/REDS/NAFNet-width64.yml) \| [test](./options/test/REDS/NAFNet-width64.yml)|
|NAFSSR-L_4x|Flickr1024|24.17|0.7589|[gdrive](https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1P8ioEuI1gwydA2Avr3nUvw?pwd=qs7a)|[train](./options/test/NAFSSR/NAFSSR-L_4x.yml) \| [test](./options/test/NAFSSR/NAFSSR-L_4x.yml)|
|NAFSSR-L_2x|Flickr1024|29.68|0.9221|[gdrive](https://drive.google.com/file/d/1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1GS6YQSSECH8hAKhvzw6GyQ?pwd=2v3v)|[train](./options/test/NAFSSR/NAFSSR-L_2x.yml) \| [test](./options/test/NAFSSR/NAFSSR-L_2x.yml)|
|Baseline-GoPro-width32|GoPro|32.4799|0.9575|[gdrive](https://drive.google.com/file/d/14z7CxRzVkYEhFgsZg79GlPTEr3VFIGyl/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1WnFKYTAQyAQ9XuD5nlHw_Q?pwd=oieh)|[train](./options/train/GoPro/Baseline-width32.yml) \| [test](./options/test/GoPro/Baseline-width32.yml)|
|Baseline-GoPro-width64|GoPro|33.3960|0.9649|[gdrive](https://drive.google.com/file/d/1yy0oPNJjJxfaEmO0pfPW_TpeoCotYkuO/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1Fqi2T4nyF_wo4wh1QpgIGg?pwd=we36)|[train](./options/train/GoPro/Baseline-width64.yml) \| [test](./options/test/GoPro/Baseline-width64.yml)|
|Baseline-SIDD-width32|SIDD|39.8857|0.9596|[gdrive](https://drive.google.com/file/d/1NhqVcqkDcYvYgF_P4BOOfo9tuTcKDuhW/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1wkskmCRKhXq6dGa6Ns8D0A?pwd=0rin)|[train](./options/train/SIDD/Baseline-width32.yml) \| [test](./options/test/SIDD/Baseline-width32.yml)|
|Baseline-SIDD-width64|SIDD|40.2970|0.9617|[gdrive](https://drive.google.com/file/d/1wQ1HHHPhSp70_ledMBZhDhIGjZQs16wO/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1ivruGfSRGfWq5AEB8qc7YQ?pwd=t9w8)|[train](./options/train/SIDD/Baseline-width64.yml) \| [test](./options/test/SIDD/Baseline-width64.yml)|


### Image Restoration Tasks 

| Task                                 | Dataset | Train/Test Instructions            | Visualization Results                                        |
| :----------------------------------- | :------ | :---------------------- | :----------------------------------------------------------- |
| Image Deblurring                     | GoPro   | [link](./docs/GoPro.md) | [gdrive](https://drive.google.com/file/d/1S8u4TqQP6eHI81F9yoVR0be-DLh4cNgb/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1yNYQhznChafsbcfHO44aHQ?pwd=96ii)|
| Image Denoising                      | SIDD    | [link](./docs/SIDD.md)  | [gdrive](https://drive.google.com/file/d/1rbBYD64bfvbHOrN3HByNg0vz6gHQq7Np/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1wIubY6SeXRfZHpp6bAojqQ?pwd=hu4t)|
| Image Deblurring with JPEG artifacts | REDS    | [link](./docs/REDS.md)  | [gdrive](https://drive.google.com/file/d/1FwHWYPXdPtUkPqckpz-WBitpVyPuXFRi/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/17T30w5xAtBQQ2P3wawLiVA?pwd=put5) |
| Stereo Image Super-Resolution | Flickr1024+Middlebury    | [link](./docs/StereoSR.md)  | [gdrive](https://drive.google.com/drive/folders/1lTKe2TU7F-KcU-oaF8jqgoUwIMb6RW0w?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1kov6ivrSFy1FuToCATbyrA?pwd=q263 ) |
-->

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