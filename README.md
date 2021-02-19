# Do Generative Models Know Disentanglement? Contrastive Learning is All You Need

> **Do Generative Models Know Disentanglement? Contrastive Learning is All You Need** <br>
> Xuanchi Ren*, Tao Yang*, Yuwang Wang and Wenjun Zeng <br>
> *arXiv preprint arXiv:2007.06600*<br>
> \* indicates equal contribution 
> 
[[Paper]()]
[[Appendix]()]
[[Demo]()]

In this repo, we propose an unsupervised and model-agnostic method: Disentanglement via Contrast (DisCo) in the Variation Space.
This code discovers disentangled directions in the latent space and extract disentangled representations from images with Contrastive Learning.
DisCo achieves the state-of-the-art disentanglement given pretrained non-disentangled generative models, including GAN, VAE, and Flow. 

**NOTE:** The following results are obtained in a completely *unsupervised* manner.

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

| Shapes3D | |
| :---: | :---: |
| Wall Color | Floor Color |
| ![image](./images/shape3d/style_shape_back.png) | ![image](./images/shape3d/style_shape_floor.png) |
| Object Color | Pose |
| ![image](./images/shape3d/style_shape_object.png) | ![image](./images/shape3d/style_shape_pose.png) |


## BibTeX

```bibtex
@article{shen2020closedform,
  title   = {Closed-Form Factorization of Latent Semantics in GANs},
  author  = {Shen, Yujun and Zhou, Bolei},
  journal = {arXiv preprint arXiv:2007.06600},
  year    = {2020}
}
```
