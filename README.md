# IDRiD_sub1_EX
2018 the IDRiD challenge sub1(hard exudates segmentation) using unet based on tensorflow and tensorlayer



## task description

> from: https://idrid.grand-challenge.org/Segmentation/ 

 The purpose of this challenge is to directly compare the methods developed for automatic segmentation of retinal lesions. 

 **Background** 

 Diabetic Retinopathy is referred as a clinical diagnosis, depicted by the presence (see Fig.) of one or more several retinal lesions like microaneurysms (MA), hemorrhages (HE), hard exudates (EX), and soft exudates (SE). 

![Diabetic Retinopathy](https://github.com/anonymoussss/IDRiD_sub1_EX/blob/master/assets/1.png?raw=true)



Segmentation of individual or multiple lesion associated with diabetic retinopathy. The sub-challenge can be divided in four different tasks; participants can submit results for at least one of the following tasks:

1. Microaneurysms Segmentation
2. Hemorrhage Segmentation
3. Soft Exudates Segmentation
4. Hard Exudates Segmentation

**Task**

 Segmentation of individual or multiple lesion associated with diabetic retinopathy. The sub-challenge can be divided in four different tasks; participants can submit results for at least one of the following tasks: 

1. Microaneurysms Segmentation
2. Hemorrhage Segmentation
3. Soft Exudates Segmentation
4. Hard Exudates Segmentation

**Data**

 It consists of 81 images with pixel level annotation for different abnormalities like MA, HE, SE and EX. 

| TRAINING SET | TESTING SET |
| ------------ | ----------- |
| MA - 54      | MA - 27     |
| EX - 54      | EX - 27     |
| HE - 53      | HE - 27     |
| SE - 26      | SE - 14     |

In this repository, I just use the "EX" class for simplicity, actually  I use only one type of data to train at a time, and The data will be placed in the following format.

```python
├── Data
│   ├── images
│       ├── training
│       ├── validation
│   ├── annotations
│       ├── training
│       ├── validation
```

The data needs to go to the competition website to download ( https://idrid.grand-challenge.org/Data_Download/ ),  I just show one sample for example, the left one is the original image, the right one is the corresponding groud truth.

![data](https://github.com/anonymoussss/IDRiD_sub1_EX/blob/master/assets/2.png?raw=true)

 **Result Submission** 

 Intensity image with particular abnormality segmented as foreground and rest part of an image as background. It will be utilized to evaluate Sensitivity (SN), Specificity (SP) and Positive Predictive Value (PPV) at different threshold values. Participants may submit results for multiple lesions but they have to submit for each lesion separately. Note: While submission, upload gray-scale images (in .jpg format without changing the original file name) obtained using your method described in the short paper. 

 **Performance Evaluation** 

 This challenge evaluates the performance of the algorithms for lesion segmentation using the available binary masks. It is done by computing area under Precision (Positive Predictive Value) and Sensitivity (Recall) curve. The curve is obtained by thresholding the results at 33 equally spaced instances. i.e. [0,0.03125,0.0625,…,1]. The area under precision-recall (AUPR) is used to obtain a single score. 

> In this section, The area under precision-recall (AUPR) needs you to know what is precision recall curve (PRC), about the concepts of PRC， I write it in my blog,  plz see: https://anonymoussss.github.io/2018/07/31/2x2-Tables-SN-SP-PPV-NPV-OR-RR/ 



## Train

- first of all, change your label images to 0-1 binary mask (eg. 0 for background and 1 for foreground)

- then just run

```
python unet.py
```



## Result

the model effect  will be shown at each epoch by tensorboard , including the train curves, test curves, the oiginal image, corresponding  gt and prediction.

**pixel auccurcy**

![pixel auccurcy](https://github.com/anonymoussss/IDRiD_sub1_EX/blob/master/assets/3.png?raw=true)

**AUPR on test datasets** 

![AUPR on test datasets ](https://github.com/anonymoussss/IDRiD_sub1_EX/blob/master/assets/4.png?raw=true)

**from left to right: original image, gt image, prediction image** 

![prediction comparation](https://github.com/anonymoussss/IDRiD_sub1_EX/blob/master/assets/5.png?raw=true)



**final result**

| AUC   | PRC  |
| ----- | ---- |
| 0.975 | 700  |

