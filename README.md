# image_segmentation_chromosomes

This repository uses machine learning for the computer vision problem of image segmentation to distinguish between overlapping chromosomes.  This problem was proposed by [Artificial Intelligence Open Networks (AI ON)](http://ai-on.org/projects/visual-segmentation-of-chromosomal-preparations.html), copied below for convenience.  

## Problem description

In cytogenetics, experiments typically starts from chromosomal preparations fixed on glass slides. Occasionally a chromosome can fall on another one, yielding overlapping chromosomes in the image. Before computers and images processing with photography, chromosomes were cut from a paper picture and then classified (at least two paper pictures were required when chromosomes are overlapping). More recently, automatic segmentation methods were developped to overcome this problem. Most of the time these methods rely on a geometric analysis of the chromosome contour and require some human intervention when partial overlap occurs. Modern deep learning techniques have the potential to provide a more reliable, fully-automated solution.
Why this problem matters

A fast and fully-automated segmentation solution can allow to scale certain experiments to very large number of chromosomes, which was not possible before. E.g. quantitative analysis of hybridization fluorescent signal on metaphasic chromosomes in the case of telomere length analysis.
Datasets

    overlapping-chromosomes - note that following this link triggers a download.

## References

    A Geometric Approach To Fully Automatic Chromosome Segmentation
    Automated Discrimination of Dicentric and Monocentric Chromosomes by Machine Learning-based Image Processing
    An Efficient Segmentation Method for Overlapping Chromosome Images
    A Review of Cytogenetics and its Automation

	
## Data

The data set is comprised of 13,434 grayscale images (94 x 93 pixels) of overlapping chromosomes. For each image, there is a ground truth segmentation map of the same size, shown below. In the segmentation map, class labels of 0 correspond to the background, class labels of 1 correspond to non-overlapping regions of one chromosome, class labels of 2 correspond to non-overlapping regions of the second chromosome, and labels of 4 correspond to overlapping regions. 

![input](/images/input_segmentation.png)

In terms of data preprocessing, a few erronous labels of 4 were corrected to match the label of the surrounding pixels. Misslabels on the non-overlapping regions, which were seen as artifacts in the segmentation map, were addressed by assigning them to the background class unless there were at least three neighbouring pixels that were in the chromosome class. The images were cropped to 88 x 88 pixels to use pooling layers with stride of two.

![preprocessing](/images/input_correction.png)
	
## Methodology

One simple solution is to classify pixels based on their intensity. Unfortunately, when histograms of the overlapping region and the single chromosome regions are plotted below, there is significant overlap between the two histograms. Thus, a simple algorithm based on a threshold pixel intensity value would perform poorly.

![hist](/images/histogram.png)

The deep learning solution used for this problem was inspired by [U-net](https://arxiv.org/abs/1505.04597v1) (shown below), a convolutional neural network for image segmentation that was demonstrated on medical images of cells.

![unet](/images/unet.png)

A convolutional neural network was created for this problem (see below). The model is designed so that the output segmentation has the same length and width as the input image. To reduce computation time and storage, the model was also simplified, with almost a third fewer layers and blocks. This is because the dimensions of the input image are small (an order of magnitude smaller than the input to U-Net) and thus too many pooling layers is undesirable. Furthermore, the set of potential objects in the chromosome images is small and the set of potential chromosome shapes is small, which reduces the scope of the problem and thus the modeling needs.

![overlappingseg](/images/overlapsegmentationnet.png)

Various hyperparameters of the model were tested, included encoding the class labels as integers, using one-hot encodings, combining the classes of the non-overlapping regions, treating each chromosome seperately, using or not using class weights, trying different activation functions, and choosing different loss functions.

The model is trained on the first 80% of the data (10,747 samples) and tested on the last 20% of the data (2,687 samples).

## Results

The model is able to achieve an intersection over union (IOU) of 92%, which corresponds to a dice score of 96%. The IOU of each of the chromosomes, as well as the overlapping region, is shown below, along with graphs of IOU and loss versus epoch, and sample output. 

![resultstable](/images/resultstable.png)

![IOUlossgraph](/images/IOUlossgraph.png)

![sampleoutput](/images/sampleoutput.png)

