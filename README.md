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

| Cars3D | Chairs | CelebA |
| :---: | :---: | :---: |
| ![image](./images/car_ours_1.png) | ![image](./images/chairs_ours_1.png) | ![image](./images/celeba_ours_1.png) |

| Cat | Anime | 
| :---: | :---: |
| ![image](./images/cat_1.png) | ![image](./images/anima_1.jpg) |

| Market-1501 | 
| :---: | 
| ![image](./images/reid_2.png) |

| Celeba | | |
| :-- | :-- | :-- |
| ![image](./images/movie_3.gif) | ![image](./images/movie_4.gif) | ![image](./images/movie_5.gif)
| ![image](./images/movie_6.gif) | ![image](./images/movie_7.gif) | ![image](./images/movie_8.gif)
    

## BibTeX

```bibtex
@article{shen2020closedform,
  title   = {Closed-Form Factorization of Latent Semantics in GANs},
  author  = {Shen, Yujun and Zhou, Bolei},
  journal = {arXiv preprint arXiv:2007.06600},
  year    = {2020}
}
```
