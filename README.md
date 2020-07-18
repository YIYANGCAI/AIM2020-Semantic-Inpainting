# AIM2020-Inpainting-Track-II
ECCV AIM2020 Inpainting with semamtic guidance

## Introduction

This is my first deep learning based task competition on computer vision
My task is inpainting with semantic guidance.
More information can be found on the [offcial website](https://competitions.codalab.org/competitions/24676)

## Usage

I use semantic-edge merged labels for my model's guidance, which can make inpainting on instances and edges better
To improve the perforamnce of the model on PSNR SSIM, we use a multi-scaled merge method. Since low resolution's inpainting's color is better and high resolution's structure is better, when we process one single image, it is resized into 128, 256, 512 types and get the final results from three outputs' linear combination.
```
outputs = outputs1_origin * w_1 + outputs2_origin * w_2 + outputs3_origin * w_3
```
There are still a lot of issue to be done.