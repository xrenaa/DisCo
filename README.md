# Do Generative Models Know Disentanglement? Contrastive Learning is All You Need

<a href="https://arxiv.org/abs/2102.10543"><img src="https://img.shields.io/badge/arXiv-2102.10543-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

> **Do Generative Models Know Disentanglement? Contrastive Learning is All You Need** <br>
> Xuanchi Ren*, Tao Yang*, Yuwang Wang and Wenjun Zeng <br>
> *arXiv preprint arXiv:2102.10543*<br>
> \* indicates equal contribution 
> 
[[Paper](https://arxiv.org/abs/2102.10543)]
[[Appendix](https://xuanchiren.com/pub/DisCo_appendix.pdf)]


## Recent Updates
**`2021.2.24`**: Add Appendix.   
**`2021.5.01`**: Plan to release code.


## Description   
![image](./images/DisCo_overview_crop.png)

In this repo, we propose an **unsupervised** and **model-agnostic** method: Disentanglement via Contrast (**DisCo**) in the Variation Space.
This code **discovers disentangled directions** in the latent space and **extract disentangled representations** from images with **Contrastive Learning**.
DisCo achieves the state-of-the-art disentanglement given pretrained non-disentangled generative models, **including GAN, VAE, and Flow**.  


**NOTE:** The following results are obtained in a completely *unsupervised* manner. More results (including VAE and Flow) are presented in *Appendix*.

## Disentangled Directions in the Latent Space
| FFHQ StyleGAN2 |  |
| :---: | :---: |
| Pose | Smile |
| ![image](./images/FFHQ/FFHQ_pose.png) | ![image](./images/FFHQ/FFHQ_smile.png) |
| Race | Oldness |
| ![image](./images/FFHQ/FFHQ_color.png) | ![image](./images/FFHQ/FFHQ_old.png) |
| Overexpose | Hair |
| ![image](./images/FFHQ/FFHQ_over.png) | ![image](./images/FFHQ/FFHQ_hair.png) |

| Shapes3D StyleGAN2 |  |
| :---: | :---: |
| Wall Color | Floor Color |
| ![image](./images/shape3d/style_shape_back.png) | ![image](./images/shape3d/style_shape_floor.png) |
| Object Color | Pose |
| ![image](./images/shape3d/style_shape_object.png) | ![image](./images/shape3d/style_shape_pose.png) |

| Car3D StyleGAN2 | |
| :---: | :---: |
| Azimuth | Yaw |
| ![image](./images/car3d/style_car_azi.png) | ![image](./images/car3d/style_car_yaw.png) |

| Anime SNGAN | |
| :---: | :---: |
| Pose | Natureness |
| ![image](./images/Anime/SN_Aime_appendix_pose.png) | ![image](./images/Anime/SN_Aime_appendix_nature.png) |
| Glass | Tone |
| ![image](./images/Anime/SN_Aime_appendix_glass.png) | ![image](./images/Anime/SN_Aime_appendix_hair.png) |

## Disentangled Representation
| Shapes3D | |
| :---: | :---: |
| MIG | DCI |
| ![image](./images/distribution_mig.png) | ![image](./images/distribution.png) |

**NOTE:** DisCo achieves the state-of-the-art disentanglement


## BibTeX

```bibtex
@article{ren2021DisCo,
  title   = {Do Generative Models Know Disentanglement? Contrastive Learning is All You Need},
  author  = {Xuanchi Ren, Tao Yang, Yuwang Wang, Wenjun Zeng},
  journal = {arXiv preprint arXiv: 2102.10543},
  year    = {2021}
}
```

