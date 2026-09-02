# Data Augmentation Approaches to Improve Accuracy, Precision, and Recall When Training Models on Image-Based Weed Datasets

**COMPSCI 4Z03 – Directed Readings**
**Michele Akeson**
**400027379**
**August 14, 2024**

---

## Table of Contents

* [Abstract](#abstract)
* [Introduction](#introduction)
* [Data Augmentation](#data-augmentation)

  * [Geometric Transformation](#geometric-transformation)
  * [Colour Transformation](#colour-transformation)
  * [Random Erasing](#random-erasing)
* [Ablation Study](#ablation-study)

  * [Methodology](#methodology)
  * [Results](#results)
  * [Control](#control)
  * [Flipping](#flipping)
  * [Colorless](#colorless)
* [Conclusion](#conclusion)
* [References](#references)

---

## Abstract

Managing the impact of weeds on crop yield is a crucial area of focus in the agriculture space due to the significant reduction weeds have on overall production. Traditional weed management methods such as synthetic herbicides and mechanical interventions are available but impose environmental impacts with declining long-term efficacy due to weed resistance, or harm to soil quality. AI-based weed detection utilizing deep learning and computer vision offers a promising alternative in weed management, but faces challenges due to the visual similarity between weeds and crops, occlusion commonly viewed in image capture of weeds and plants, and limited dataset variability and size currently available to the public.

This paper evaluates the effectiveness of data augmentation techniques, specifically geometric transformation, colour transformation, and random erasing, and their individual capacity to improve the performance of weed detection models. Geometric transformation (flipping, rotating, or translating) and random erasing demonstrate notable improvements in model accuracy by reducing overfitting and enhancing dataset diversity. However, colour transformation results vary, depending on the image data type and species characteristics. Moreover, this paper highlights the importance of diverse and innovative data augmentation strategies to enhance weed detection models' accuracy, precision, and recall, and the need for larger datasets. Future advancements in AI-based weed management require comprehensive, varied datasets and sophisticated augmentation techniques for sustainable and efficient agricultural solutions.

Additionally, this paper conducts an ablation study comparing the impact each of the above augmentation techniques has on accuracy, precision, and recall in image classification of weeds in a small dataset.

---

# Introduction

Weeds pose significant impact to crop yield, often considered more detrimental to production than insects, fungus, or plant disease, reducing production between 15–76% depending on crop type, soil quality, and location [1]. As a consequence, weed management is an essential area of focus in the agriculture space [3]. Contemporary methods exist to mitigate the impact of weeds on crop yield, typically categorized as either mass application of synthetic herbicides, or mechanical intervention (tilling, manual weeding, or introduction of a weed’s natural predator) [2]. Both approaches, however, impose significant environmental drawbacks which heavily limit their feasibility as long-term weed management solutions.

Synthetic herbicide, for instance, has serious negative impacts on the ecosystem, while also becoming increasingly less effective due to evolution of herbicide-resistant plants and more restrictive regulations for herbicide use [3]. Large-scale mechanical intervention, such as tilling, has been reported to significantly contribute to greenhouse gas emissions and reduce carbon storage in soil, reducing soil quality over time [4]. Through this lens, it is evident that new weed management strategies are required to mitigate the impact of weeds on crop yield without imposing negative effects towards soil quality and the environment, as caused by traditional tactics.

Weed detection is the automatic classification and/or object detection of weeds, often achieved through AI-based computer vision. Due to recent advancements in deep learning techniques, weed detection has become an important tool in weed management [5]. The current use of weed detection alongside ancillary technologies, such as IoT, UAVs, and field robotics, provides clear indication of growing popularity, and more importantly, demonstrates improved precision weed control [6], with some systems achieving as high as 92.9% accuracy when detecting weeds [8]. Although weed detection appears to be a crucial aspect of addressing the demand for an alternative weed management strategy, several difficult challenges impact the generalizability of weed detection models [14], hampering their capacity to become commercially viable.

A significant challenge is the complexity of identifying weeds in contrast to crop plants within supervised learning models. This complexity can be attributed to the following: weeds and crop plants may present similar features, such as shape, colour, or texture at different growth stages; plants impose occlusion, blocking surrounding plants during data capture; and finally, plants display different texture or colour due to lighting conditions [5].

Moreover, as a result of high variability between weed and crop plant species, weed detection models may require development on a per-species basis [7], making commercial application of this tool more difficult. This challenge becomes more pronounced as image-based data is far more prominent compared to other sensor-based data in weed datasets. That is to say: ultrasound, light detection and ranging (LiDAR), and optoelectronic sensors are used less frequently in weed datasets compared to image capture [7]. The most notable forms of image-based data come in the form of RGB image capture, due to relatively low camera cost, and multispectral imaging, such as near infrared (NIR) [7].

Overfitting, characterized by a model matching training data too closely, resulting in poor performance when predicting against new, unseen data, is a major risk in deep learning. The risk of overfitting significantly increases when relying on small datasets with limited diversity [7].

It is here that another major challenge for weed detection can be observed. A literature review conducted by [5] evaluated 18 existing weed and crop datasets, concluding that there is a lack of large datasets required to adequately train weed detection models. They further postulate the necessity to assemble larger datasets with variations in the type of weeds and crop plants, weather conditions (i.e. lighting conditions), and growth stages.

A more recent literature review conducted by [7] evaluated 13 publicly available datasets, six of which overlap with datasets reviewed by [5]. This review established that, while each dataset provides useful imagery and data annotation, a lack of consistency, as well as image capture under limited scenarios of lighting and growth stage, prompt the need for larger-scale datasets.

Due to the complexity of distinguishing between weeds and crop plants, as well as the limitations of publicly available datasets, the following question must be posed: what actions can be taken to improve the accuracy, precision, and recall of weed detection models, given the above constraints?

One such solution is to use data augmentation; a suite of techniques that artificially enhance the size, quality, and variation of training datasets. Data augmentation tackles the issue of overfitting within the context that more examples and features can be extracted from the original dataset by manipulating that data via either data warping or oversampling [9]. Data augmentation is an explicit form of regularization, which is any mechanism that reduces overfitting, frequently used in training convolutional neural networks [14].

This paper examines several articles in the evaluation of existing and newly introduced data augmentation techniques in the space of weed detection. Specifically, this paper will define and assess several approaches categorized under data warping: geometric transformation, colour transformation, and random erasing. Each technique will be reviewed via its effect on accuracy, precision, and recall when applied to weed datasets in an effort to establish effective data augmentations that should be applied to image-based weed datasets.

# Data Augmentation

Data augmentations artificially inflate the size of a training dataset through either data warping or oversampling [9]. Augmentations within the domain of data warping transform data without altering the labels of that data.

For image-based data, this can be observed in:

* **Geometric transformation:** flipping, translating, or rotating image data
* **Colour transformation:** altering the colour channels of RGB images, adjusting colour to grayscale, or adjusting intensity of colour to be within a specific range
* **Cropping:** removing segments of an image connecting to an outer border
* **Kernel filtering:** a technique in image processing to sharpen and blur images
* **Random erasing:** erasing random segments within an image

Each of these techniques may assist in improving existing weed datasets and reducing overfitting for future weed detection models.

## Geometric Transformation

Geometric transformation, sometimes referred to as spatial transformation, covers the image-based data augmentation approaches of flipping, rotating, or translating image data. This form of data augmentation is considered relatively safe in regard to preserving labels, while also being easy to implement [9].

Flipping images along the horizontal or vertical axis is an easily applied augmentation to datasets; however, it may not preserve labels, such as in the case of MNIST (i.e. tasks with written symbols or digits) [9].

Rotation involves rotating an image between 1–359 degrees, while translation moves an image in any direction, filling in remaining space with constant values between 0 and 255 within the colour space [9]. Translation can be an especially helpful augmentation to prevent models from overfitting to images that have perfectly centered objects.

A study conducted by [10] investigated the impact of individual data augmentation transformations on the semantic segmentation (i.e. pixel-level classification) of crops and weeds compared to raw data. Within this experiment, the Crop/Weed Field Image Dataset (CW-FID), which includes pixel-level annotations for 60 carrot and weed images, was used. Each image had a resolution of 1296 × 966 pixels and was captured under a single lighting condition.

Several geometric transformations were applied to the dataset, including, but not limited to:

* Horizontal flipping
* Vertical flipping
* Rotation
* Scaling

The individual and combined impact of these transformations was evaluated using the Dice Similarity Coefficient (DSC), a metric that measures how well two sets overlap. Moreover, it acts as a composite metric of precision and recall into a single score.

This study concluded that each augmentation improved performance relative to the raw data. However, the combination of all augmentation tactics performed best, achieving the best overall result in the experiment: a **19.96% improvement** compared to the use of unaugmented data.

These results suggest that a combination of geometric augmentations can significantly improve dataset efficacy. As crop plants and weeds display additional features, such as colour, other augmentation strategies may pose possible advantages.

## Colour Transformation

As stated previously, the primary form of image-based data capture comprising weed datasets comes from low-cost RGB cameras. This is an important component to note as colour is a prominent characteristic of weeds and plants.

For more context, digital images typically encode data via height × width × colour channel. Thus, data augmentation via colour transformation targets the three colour channels of a pixel — Red, Green, and Blue — which store values between 0 and 255, representing the intensity of that respective colour.

Colour transformation may involve one of the following:

* Isolating a specific colour channel by setting all other channels to 0
* Using a colour histogram to change the intensity of colours within the image
* Adjusting colour intensity to address varying lighting conditions
* Converting an image to grayscale

Due to the similarities in colour between the majority of weed and crop plant species, differentiating between these two classes is difficult [5]. Some studies postulate that due to the similar colour between weed and crop plants, using colour as a feature is not valuable for weed detection [11].

A study conducted by [11] opted to perform data collection using three monochrome cameras, creating a weed dataset without colour, and a mobile canopy above the cameras to achieve consistent lighting. This study aimed to determine the impact of weed-to-grass ratios in image datasets.

The training set was divided evenly between training and testing for a CNN, determining that a 1:10 weed-to-grass ratio yielded a high true-positive rate, while a 1:20 ratio yielded a low false-positive rate. Additionally, ten different random training sets were used for comparison.

The results of this study, while not explicitly investigating the impact of the colour feature in weed detection, suggest that absence of a colour feature in a weed dataset does not impact accuracy significantly.

While this is counter to common practice, as grayscale augmentation is often shown to reduce accuracy by approximately 3% [13], these findings are further affirmed by a study implemented by [12]. This study sought to examine an encoder-decoder deep learning network in semantic segmentation of crop plants and weeds, comparing the impact of colour transformation with RGB versus NIR image capture.

Two datasets were used:

* A sugar beet image dataset
* An oil seed image dataset captured with a commercial RGB camera

The specific colour transformations included:

* Histogram equalization
* Photoshop Auto Contrast
* Deep Photo Enhancer

The results of the study showed that colour space transformation using RGB image data did not improve the segmentation results, but significantly improved the segmentation accuracy when image data included NIR information.

This indicates the effectiveness of NIR for better segmentation under poor lighting, and the value of colour transformation when augmenting data with NIR information. It also conveys that colour transformation poses no benefit with RGB image data.

Thus, colour transformation in the space of weed detection may be considered highly dependent on the form of data collection used in the dataset, such as RGB versus NIR, and the differences in colour characteristics of the weed and crop plant species respectively.

## Random Erasing

A more recent augmentation strategy is **random erasing**. This technique specifically focuses on combating the issue of occlusion faced in image detection.

Proposed in a study by [14], random erasing works by randomly selecting a rectangular segment within an image and manipulating the pixel data within that segment. This can be achieved by:

* Changing the colour channels to a value of 0
* Changing the colour channels to a value of 255
* Giving each pixel a random value

In more general terms, this augmentation approach intentionally introduces occlusion to arbitrary regions of an image, forcing a model to learn more descriptive features from examples in the dataset.

The study conducted by [14], while not explicitly focused on weed datasets, investigated the impact of random erasing on both image classification and object detection when training convolutional neural networks.

For image classification, this study utilized four datasets, most prominently CIFAR-10 and CIFAR-100, while relying on PASCAL VOC 2007 as a dataset for object detection.

The study used classification accuracy and error rates as metrics to evaluate the impact of random erasing, reporting significant improvements in accuracy percentages and reductions in error rates for various CNN architectures when utilizing random erasing as an augmentation.

Within image classification, as an example, the error rate on CIFAR-100 was reduced to **17.73%**, compared to the baseline error rate of **18.49%** without random erasing.

The study also found random erasing had a slight positive effect for object detection.

As occlusion is a consistent challenge posed in the weed detection space, this novel data augmentation approach could pose as a potential tactic to address and further improve future weed detection models by augmenting existing datasets.

# Ablation Study

An ablation study was conducted to examine the impact of two augmentation techniques on accuracy, precision, and recall when training a convolutional neural network for image classification of weed and crop plants.

In this study:

1. **Horizontal flipping** of images across the y-axis
2. **Conversion of images to grayscale**

were tested individually to determine whether use of these techniques reduced overfitting by increasing dataset size and/or variation compared to no augmentation.

## Methodology

For this study, the [Crop and Weed Detection Dataset](https://datasetninja.com/crop-weed-detection) was used for training each model.

The dataset provided two folders:

* `img/` — containing weed and crop images
* `ann/` — containing a JSON file associated with each image, indicating whether the image contains a weed or crop

The dataset contained:

| Class     | Number of Images |
| --------- | ---------------: |
| Weed      |              666 |
| Crop      |              634 |
| **Total** |        **1,294** |

Each image was **512 × 512 pixels**.

To conduct this study, the dataset had to be formatted such that it could be used by the convolutional neural network created for this study. Code was written that sorted images into one of two subdirectories, `weed/` or `crop/`, using the associated annotation file to determine which folder an image belongs to.

After sorting the images into the correct folder, augmentations were applied to create two separate datasets to be used for training.

### Flipping Dataset

When applying the flipping augmentation, all images from the `weed/` and `crop/` folders were duplicated and copied to a new folder:

```text
flippedData/
├── weed/
└── crop/
```

In addition to copying the original images, a horizontally flipped copy of each image was added, increasing the dataset for the flipping augmentation to **2,588 images**.

### Colorless Dataset

When applying the colour transformation augmentation, all images from the `weed/` and `crop/` folders were copied to a new `colorlessData/` folder and converted to grayscale.

The resulting dataset contained **1,294 grayscale images**.

### Model Configuration

The model created has 10 layers and expects an input image of size **150 × 150 pixels**. This required scaling all images in the dataset before passing each image as an input during training. This was done to reduce training time for each model.

For each model:

| Parameter                |     Value |
| ------------------------ | --------: |
| Training split           |       80% |
| Validation split         |       20% |
| Learning rate            |     0.001 |
| Epochs                   |        10 |
| Batch size               |        20 |
| Input size               | 150 × 150 |
| Classification threshold |       0.7 |

A classification threshold of `0.7` was used such that only outputs greater than 0.7 were considered positive predictions of one of the two classes.

For each model, training accuracy, precision, and recall were compared to validation accuracy, precision, and recall.

The specific implementation of the project can be viewed at:

[GitHub Repository](https://github.com/Michele-4-Akeson/4Z03-WeedDetectionML)

# Results

## Summary

| Metric        | Control — Train | Control — Validation | Flipping — Train | Flipping — Validation | Colorless — Train | Colorless — Validation |
| ------------- | --------------: | -------------------: | ---------------: | --------------------: | ----------------: | ---------------------: |
| **Accuracy**  |            0.92 |                 0.91 |             0.93 |              **0.94** |              0.95 |                   0.92 |
| **Precision** |            0.96 |                 0.93 |             0.94 |              **0.97** |              0.97 |                   0.93 |
| **Recall**    |            0.87 |                 0.89 |             0.91 |              **0.92** |              0.93 |                   0.90 |
| **Loss**      |            0.19 |                 0.33 |             0.17 |              **0.19** |          **0.10** |                   0.39 |

## Control

The control model, trained on the original dataset without any augmentation, showed sufficient performance. It achieved a training accuracy of **0.92** with a loss of **0.19**, a precision of **0.96**, and a recall of **0.87**.

The validation results were slightly lower, with an accuracy of **0.91**, a loss of **0.33**, a precision of **0.93**, and a recall of **0.89**.

These results suggest that the model was able to generalize well to unseen data, though there was a small drop in performance from training to validation, which could indicate minor overfitting.

## Flipping

Applying the horizontal flipping augmentation significantly increased the dataset size and led to noticeable improvements in model performance.

The flipped model achieved:

* Training accuracy: **0.93**
* Training loss: **0.17**
* Training precision: **0.94**
* Training recall: **0.91**

The validation performance improved as well:

* Validation accuracy: **0.94**
* Validation loss: **0.19**
* Validation precision: **0.97**
* Validation recall: **0.92**

This suggests that the flipping augmentation effectively enhanced the model's ability to generalize, reducing overfitting and improving prediction accuracy on the validation set.

## Colorless

When the dataset was converted to grayscale, the model performed even better during training, achieving an accuracy of **0.95**, with a low loss of **0.10**, a precision of **0.97**, and a recall of **0.93**.

However, this improvement did not carry over to the validation set as strongly as with the flipping augmentation.

The validation results were:

* Validation accuracy: **0.91**
* Validation loss: **0.38**
* Validation precision: **0.93**
* Validation recall: **0.90**

These results indicate that while the grayscale augmentation helped the model perform well during training, it may have caused overfitting or reduced the model's ability to generalize, as evidenced by the increased validation loss and the smaller improvement in validation accuracy.

In reviewing the metrics for each model, it is clear that compared to no augmentation, flipping provides a significant improvement in accuracy, precision, and recall for both training and validation.

The grayscale augmentation, while beneficial during training, appears to introduce challenges in generalization, suggesting that the model may rely too heavily on the simplified grayscale features, leading to overfitting.

## Results Interpretation

The results indicate that **horizontal flipping was the most effective of the two augmentation techniques tested in the ablation study**.

Compared with the control model, flipping improved validation performance as follows:

| Metric    | Control Validation | Flipping Validation | Improvement |
| --------- | -----------------: | ------------------: | ----------: |
| Accuracy  |               0.91 |            **0.94** |       +0.03 |
| Precision |               0.93 |            **0.97** |       +0.04 |
| Recall    |               0.89 |            **0.92** |       +0.03 |
| Loss      |               0.33 |            **0.19** |       -0.14 |

The flipping augmentation therefore improved all three primary evaluation metrics while also substantially reducing validation loss.

Grayscale conversion produced strong training metrics but did not provide the same validation improvement. In particular, its validation loss increased substantially compared with the control, suggesting weaker generalization despite its higher training performance.

# Conclusion

In conclusion, the necessity for innovative weed management strategies in agriculture is evident due to the limitations of traditional methods such as synthetic herbicides and mechanical interventions. These traditional methods pose significant environmental drawbacks and are becoming less effective. AI-based weed detection models, though promising, face challenges such as the visual similarity between weeds and crops, occlusion issues, and limited dataset variability.

This paper evaluated the effectiveness of various data augmentation techniques: geometric transformation, colour transformation, and random erasing, in improving weed detection model performance.

Geometric transformations, such as flipping, rotating, and translating images, significantly enhanced model accuracy by reducing overfitting and increasing dataset diversity. The combination of these techniques yielded the highest improvement in detection performance.

Colour transformation presented mixed results, heavily dependent on the type of image data used (RGB vs. NIR). This finding suggests that use of colour transformation should only be applied with RGB image data when distinct colour differences exist between crop plant and weed species.

Random erasing emerged as a valuable augmentation strategy to address occlusion, a common challenge in weed detection. By artificially introducing occlusions, this technique forces models to learn more robust and descriptive features from examples, leading to improved performance in both image classification and object detection.

Overall, employing data augmentation techniques is helpful for enhancing the performance of weed detection models. Until weed datasets meet the size and variability demands required to develop weed detection models without overfitting, data augmentation should continue to be applied in contemporary data to better support the generalizability of models, improving their practical usage.



![Accuracy Results](/results/colorless_accuracy_plot.png)
![Accuracy Results](/results/colorless_loss_plot.png)
![Accuracy Results](/results/colorless_precision_plot.png)
![Accuracy Results](/results/colorless_recall_plot.png)

![Accuracy Results](/results/control_accuracy_plot.png)
![Accuracy Results](/results/control_loss_plot.png)
![Accuracy Results](/results/control_precision_plot.png)
![Accuracy Results](/results/control_recall_plot.png)

![Accuracy Results](/results/flipped_accuracy_plot.png)
![Accuracy Results](/results/flipped_loss_plot.png)
![Accuracy Results](/results/flipped_precision_plot.png)
![Accuracy Results](/results/flipped_recall_plot.png)

# References

1. Gharde, Y., Singh, P. K., Dubey, R. P., & Gupta, P. K. (2018). Assessment of yield and economic losses in agriculture due to weeds in India. *Crop Protection*, *107*, 12–18. https://doi.org/10.1016/j.cropro.2018.01.007

2. Monteiro, A., & Santos, S. (2022). Sustainable approach to weed management: The role of Precision Weed Management. *Agronomy*, *12*(1), 118. https://doi.org/10.3390/agronomy12010118

3. Loddo, D., McElroy, J. S., & Giannini, V. (2021). Problems and perspectives in weed management. *Italian Journal of Agronomy*, *16*(4). https://doi.org/10.4081/ija.2021.1854

4. Cooper, H. V., Sjögersten, S., Lark, R. M., & Mooney, S. J. (2021). To till or not to till in a temperate ecosystem? Implications for climate change mitigation. *Environmental Research Letters*, *16*(5), 054022. https://doi.org/10.1088/1748-9326/abe74e

5. Hasan, A. S., Sohel, F., Diepeveen, D., Laga, H., & Jones, M. G. K. (2021). A survey of deep learning techniques for weed detection from images. *Computers and Electronics in Agriculture*, *184*, 106067. https://doi.org/10.1016/j.compag.2021.106067

6. Vasileiou, M., Kyrgiakos, L. S., Kleisiari, C., Kleftodimos, G., Vlontzos, G., Belhouchette, H., & Pardalos, P. M. (2024). Transforming Weed Management in sustainable agriculture with Artificial Intelligence: A systematic literature review towards weed identification and deep learning. *Crop Protection*, *176*, 106522. https://doi.org/10.1016/j.cropro.2023.106522

7. Hu, K., Wang, Z., Coleman, G., Bender, A., Yao, T., Zeng, S., Song, D., Schumann, A., & Walsh, M. (2023). Deep learning techniques for in-crop weed recognition in large-scale grain production systems: A Review. *Precision Agriculture*, *25*(1), 1–29. https://doi.org/10.1007/s11119-023-10073-1

8. Sujaritha, M., Annadurai, S., Satheeshkumar, J., Kowshik Sharan, S., & Mahesh, L. (2017). Weed detecting robot in sugarcane fields using fuzzy real time classifier. *Computers and Electronics in Agriculture*, *134*, 160–171. https://doi.org/10.1016/j.compag.2017.01.008

9. Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for Deep Learning. *Journal of Big Data*, *6*(1). https://doi.org/10.1186/s40537-019-0197-0

10. Brilhador, A., Gutoski, M., Hattori, L. T., de Souza Inacio, A., Lazzaretti, A. E., & Lopes, H. S. (2019). Classification of weeds and crops at the pixel-level using convolutional neural networks and data augmentation. *2019 IEEE Latin American Conference on Computational Intelligence (LA-CCI)*. https://doi.org/10.1109/la-cci47412.2019.9037044

11. Kounalakis, T., Malinowski, M. J., Chelini, L., Triantafyllidis, G. A., & Nalpantidis, L. (2018). A robotic system employing deep learning for visual recognition and detection of weeds in Grasslands. *2018 IEEE International Conference on Imaging Systems and Techniques (IST)*. https://doi.org/10.1109/ist.2018.8577153

12. Wang, A., Xu, Y., Wei, X., & Cui, B. (2020). Semantic segmentation of crop and weed using an encoder-decoder network and image enhancement method under uncontrolled outdoor illumination. *IEEE Access*, *8*, 81724–81734. https://doi.org/10.1109/access.2020.2991354

13. Chatfield, K., Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Return of the devil in the details: Delving deep into convolutional nets. *Proceedings of the British Machine Vision Conference 2014*. https://doi.org/10.5244/c.28.6

14. Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random erasing data augmentation. *Proceedings of the AAAI Conference on Artificial Intelligence*, *34*(07), 13001–13008. https://doi.org/10.1609/aaai.v34i07.7000
