## [WFBNet](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14155)

Code for Pacific Graphics 2020 [paper]((https://onlinelibrary.wiley.com/doi/10.1111/cgf.14155)) "Coarse to Fine: Weak
Feature Boosting Network for Salient Object Detection"

## Prerequisites

- [Python 3.5](https://www.python.org/)
- [Pytorch 1.3](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy 1.15](https://numpy.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Apex](https://github.com/NVIDIA/apex)

## Download dataset

Download the following datasets and unzip them into `data` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

## Pretrained model for backbone

Please download the [pretrained model for backbone](https://download.pytorch.org/models/resnet50-19c8e357.pth)
into `res` folder

## Training

```shell
    cd src
    python3 train.py
```

## Testing

```shell
    cd src
    python3 test.py ${epoch_you_wanna_test}
```

## Saliency maps & Trained model

- saliency
  maps: [Baidu(iydw)](https://pan.baidu.com/s/1N2dYdbWIMmOOwmTekROlaA) [Google](https://drive.google.com/file/d/1UG3LEenZ1UnFQff3EL0YL8Seohd0GdBt/view?usp=sharing)
- trained
  model: [Baidu(xrdj)](https://pan.baidu.com/s/1MNfTmWLv1bavoZ10wB94eg) [Google](https://drive.google.com/file/d/1HJhROxu4j9OUmlL0shSzr1Vkyl7ZRcAZ/view?usp=sharing)
- If you want to test using our trained model, you can just download and unzip it to folder 'src', then run

```shell     
  cd src     
  python3 test.py 33
```

## Evaluation

- To evaluate the performace, please use MATLAB to run `main.m`

```shell
    cd eval
    matlab
    main
```

## Citation

```
@article {WFBNet, 
  author = {Zhang, Chenhao and Gao, Shanshan and Pan, Xiao and Wang, Yuting and Zhou, Yuanfeng}, 
  title = {{Coarse to Fine:Weak Feature Boosting Network for Salient Object Detection}}, 
  journal = {Computer Graphics Forum}, 
  year = {2020}, 
  editor = {Eisemann, Elmar and Jacobson, Alec and Zhang, Fang-Lue}, 
  volume = {39}, 
  number = {7}, 
  publisher = {The Eurographics Association and John Wiley & Sons Ltd.}, 
  pages = {411-420}, 
  DOI = {10.1111/cgf.14155} 
}
```

Thanks to [F3Net](https://github.com/weijun88/F3Net)