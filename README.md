# Image Segmentation for Overlapping Chromosomes

This repository uses machine learning for the computer vision problem of image segmentation to distinguish between overlapping chromosomes.  This problem was proposed by [Artificial Intelligence Open Networks (AI ON)](http://ai-on.org/projects/visual-segmentation-of-chromosomal-preparations.html), copied below for convenience.  

## Problem description

In cytogenetics, experiments typically starts from chromosomal preparations fixed on glass slides. Occasionally a chromosome can fall on another one, yielding overlapping chromosomes in the image. Before computers and images processing with photography, chromosomes were cut from a paper picture and then classified (at least two paper pictures were required when chromosomes are overlapping). More recently, automatic segmentation methods were developed to overcome this problem. Most of the time these methods rely on a geometric analysis of the chromosome contour and require some human intervention when partial overlap occurs. Modern deep learning techniques have the potential to provide a more reliable, fully-automated solution.

### Why this problem matters

A fast and fully-automated segmentation solution can allow to scale certain experiments to very large number of chromosomes, which was not possible before. E.g. quantitative analysis of hybridization fluorescent signal on metaphasic chromosomes in the case of telomere length analysis.

### References

[A Geometric Approach To Fully Automatic Chromosome Segmentation](https://arxiv.org/abs/1112.4164)

[Automated Discrimination of Dicentric and Monocentric Chromosomes by Machine Learning-based Image Processing](http://biorxiv.org/content/biorxiv/early/2016/01/19/037309.full.pdf) 

[An Efficient Segmentation Method for Overlapping Chromosome Images](http://research.ijcaonline.org/volume95/number1/pxc3894861.pdf)

[A Review of Cytogenetics and its Automation](http://www.scialert.net/qredirect.php?doi=jms.2007.1.18&linkid=pdf)

	
## Data and Preprocessing

The data set is comprised of 13,434 grayscale images (94 x 93 pixels) of overlapping chromosomes. For each image, there is a ground truth segmentation map of the same size, shown below. In the segmentation map, class labels of 0 (shown as black below) correspond to the background, class labels of 1 (shown as red below) correspond to non-overlapping regions of one chromosome, class labels of 2 (show as green below) correspond to non-overlapping regions of the second chromosome, and labels of 3 (shown as blue below) correspond to overlapping regions. 

![input](/images/input_segmentation.png)

In terms of data preprocessing, a few erroneous labels of 4 were corrected to match the label of the surrounding pixels. Mislabels on the non-overlapping regions, which were seen as artifacts in the segmentation map, were addressed by assigning them to the background class unless there were at least three neighbouring pixels that were in the chromosome class. The images were cropped to 88 x 88 pixels to use pooling layers with stride of two.

![preprocessing](/images/input_correction.png)
	
## Methodology

One simple solution is to classify pixels based on their intensity. Unfortunately, when histograms of the overlapping region and the single chromosome regions are plotted below, there is significant overlap between the two histograms. Thus, a simple algorithm based on a threshold pixel intensity value would perform poorly.

![hist](/images/histogram.png)

The deep learning solution used for this problem was inspired by [U-Net](https://arxiv.org/abs/1505.04597v1) (shown below, image taken from the paper), a convolutional neural network for image segmentation that was demonstrated on medical images of cells. 

![unet](/images/unet.png)

A convolutional neural network was created for this problem (see below). The model was designed so that the output segmentation map has the same length and width as the input image. To reduce computation time and storage, the model was also simplified, with almost a third fewer layers and blocks. This is because the dimensions of the input image are small (an order of magnitude smaller than the input to U-Net) and thus too many pooling layers is undesirable. Furthermore, the set of potential objects in the chromosome images is small and the set of potential chromosome shapes is small, which reduces the scope of the problem and thus the modeling needs. Also, cropping was not done within the network and padding was set to be 'same'. This was because given the small input image, it was undesirable to remove pixels. 

![overlappingseg](/images/overlapsegmentationnet.png)

Various hyperparameters of the model were tested, included encoding the class labels as integers, using one-hot encodings, combining the classes of the non-overlapping regions, treating each chromosome separately, using or not using class weights, trying different activation functions, and choosing different loss functions.

The model was trained on the first 80% of the data (10,747 samples) and tested on the last 20% of the data (2,687 samples).

## Results

To quantitatively assess the results, the intersection over union (IOU, or Jaccard's index) is calculated. IOU is a commonly reported metric for image segmentation. It is defined as the area of overlap (between the prediction and the ground truth) divided by the area of union (between the prediction and the ground truth). The image below illustrates this definition. The closer the IOU is to 100%, the better the segmentation. The model is able to achieve an IOU of 94.7% for the overlapping region, and 88.2% and 94.4% on the two chromosomes. This corresponds to dice scores of 97.3%, 93.7% and 97.1% respectively. The Dice score is an alternative metric similar to IOU, The formula for the dice score is two times the area of overlap divided by the sum of the two areas. To convert between IOU (J) and Dice (D), we have J = D/(2-D) and D = 2J(1+J).

![IOUdef](/images/IOU.png)


Graphs of IOU and loss versus epoch are shown below, along with sample predictions. Given that the testing loss is plateauing and not yet increasing, I was not worried about overfitting at this training time. 

![IOUlossgraph](/images/quantresultsplot.png)

![sampleoutput](/images/sampleoutput.png)

In terms of next steps, the data set can be supplemented with images of single chromosomes and more than two overlapping chromosomes. Data augmentation can also include transformations such as rotations, reflections, and stretching. Additional hyperparameters can also be explored, such as sample weights, filter numbers, and layer numbers. Increasing convolution size may improve misclassification between the red and green chromosomes. For upsampling, instead of cropping layers, the decoder can use pooling indices computed in the max-pooling step of the corresponding encoder, as in Segnet. 

To build a production system that can operate on entire microscope images, this model can be combined with an object detection algorithm. First, the object detection algorithm can draw bounding boxes around chromosomes in an image. Then, an image segmentation algorithm, based on the model presented here, can identify and separate chromosomes. 


