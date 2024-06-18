# Stepwise Intervention and Reweighting (SIR)

## Get Started

1. To install the necessary packages, run the following command-line code.
```
pip install -r requirements.txt
```

2. Download your dataset and update `data_path` in `config.py` to the path where you have stored the dataset. The WHDLD dataset can be downloaded [here](https://sites.google.com/view/zhouwx/dataset) and the Vaihingen dataset can be downloaded [here](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx).

3. Modify `net_name` and `pretrain_name` in the `config.py` file, then execute `train.py` to begin the training process.

4. Modify `loading_checkpoint_path` in the `config.py` file, then execute `test.py` to begin the test process.

## Useful Links

- The pre-trained models and weights can be downloaded from [torchvision](https://pytorch.org/vision/stable/_modules/torchvision.html). Most of the data preprocessing and model codes in this repository are adapted from there.
- More baseline models and datasets can be found at [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), including codes for dataset loading and preprocessing, as well as additional evaluation metrics.
- `utils/evaluate_metric.py` is adapted from [Deep-Learning-Interview-Book](https://github.com/amusi/Deep-Learning-Interview-Book/).

## References

> Z. Shao, K. Yang, and W. Zhou, “Performance evaluation of single-label and multi-label remote sensing image retrieval using a dense labeling dataset,” Remote Sensing, vol. 10, no. 6, p. 964, 2018.
> 
> Z. Shao, W. Zhou, X. Deng, M. Zhang, and Q. Cheng, “Multilabel remote sensing image retrieval based on fully convolutional network,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 13, pp. 318–328, 2020.
>  
> M. Cramer, “The dgpf-test on digital airborne camera evaluation overview and test design,” Photogrammetrie-Fernerkundung-Geoinformation, pp. 73–82, 2010.
> 
> M. Contributors, “MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark,” https://github.com/open-mmlab/mmsegmentation, 2020.
> 
> J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3431–3440.
> 
> O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. Springer, 2015, pp. 234–241.
> 
> L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 801–818.
> 
> K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in International Conference on Learning Representations, 2015.
> 
> K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.
> 
> M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International conference on machine learning. PMLR, 2019, pp. 6105–6114.