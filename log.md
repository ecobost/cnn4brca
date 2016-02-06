# Activity Log
Written by: Erick Cobos T. (a01184587@itesm.mx)

Log with activities and questions arising during every week

## Jan 25 - Feb 11
### Activities
* Writing final draft of Chapter 2
* Taking last important design decisions

### To do
* Write chapter 3 (Solution Model)
* Ready database
* Install TensorFlow (CTS/Laptop)
* Write network in Tensorflow/Keras
* Ask for a computer in A3-401 or somewhere else (maybe ask Dr. Garrido)
* Ask for institutional email

### Questions
1. Which post-processing should I use? Gaussian smoothing, cluster-based enhancement, fully connected CRFs or a combination?
2. Which evaluation metric should I use? Accuracy, F1-score, PRAUC or ROC?


----------------------------- 
## Jul 2 - Jul 8
### Activities
* Write emails looking for more digital mammograms
* Document decisions on how to crop big images.

### To do
* Ask for institutional email.

### Questions
1. 

## Jun 24 - Jul 1
### Activities
* Look for databases and its features
* Decide how to crop the images from the entire mammograms

### To do
* Write the features of the database and its labelling.
* Decide how to obtain the small crops from the big mammograms.
* Write code to automatically obtain the small crops from the mammogram.
* Update LaTex template to thesis template
* Choose software (probably Caffe)

### Questions
1. Should I try to get more digital mammograms or just go with film?.
	Answer: Enough digital mammograms, work with wath you have. If needed, ask Dr. Tamez.

## Jun 18 - Jun 23
### Activities
* Ended review
* Write ConvNet for Breast Cancer
* Rewrite some parts of proposal (introduction, objectives, methodology)

### To do
* Investigate and write the features of the database and its labelling.
* Decide how to obtain the small crops from the big mammograms
* Write software to automatically obtain the small crops from the mammogram
* Update LaTex template to thesis template
* Choose software (probably Caffe)

### Questions
1. Should i put mass vs nonmass, microcalc vs nonmicrocalc, or put every lession together (mass, microcalc, distortions, etc.) vs nonlession?. Thus, only train one network that differentiates all lessions vs no lession?. 
2. Should I use a single network with multiple outputs to classify every kind of lession?.
3. Should I use data augmentation only on the minority classes (lessions)?


## Jun 10 - Jun 17
### Activities
* Write ConvNet for Breast Cancer
* Some preprocessing experiments

### To do
* Rewrite Methodology with new experiments
* Update LaTex template to thesis template
* Investigate and write the features of the database and its labelling.
* Choose software (probably Caffe)

### Questions
1. Is the unbalanced data thing needed or does the network learns by its own?. May i be overkilling it? 
	Answer: Train normally, cross-validate the threshold

## Jun 3 - Jun 10
### Activities
* Write PracticalDL section

### To do
* Write ConvNet for Breast Cancer
* Rewrite Methodology with new experiments
* Update LaTex template to thesis template
* Investigate and write the features of the database and its labelling.
* Choose software (probably Caffe)

### Questions
1. Use NAG or SGD+Momentum?
	Answer: NAG

2. Use Bioinformatics account or create another one?.
	Answer: Bioinformatics

3. Is there a standard way to report convolutional network architectures (Krizhevsky style or Karpathy style or a table as in Striving for simplicity)?.
	Answer: No. Image if small, Table if big.

## May 27- Jun 2
### Activities
* Read CS231n
* Write Convnet section in thesis

### To do
* End writing Background
	* Practical Deep Learning
	* ConvNet for Breast Cancer
* Rewrite Methodology with new experiments
* Update LaTex template to thesis template
* Investigate and write the features of the database and its labelling.
* Choose software (probably Caffe)
* Select exactly what experiments will be run and what hyperparameters be crossvalidated

### Questions
1. Naming: Should I use loss or cost function?.
	Answer: Loss

2. How to obtain the small training images from the big images. Random sampling, crop without overlapping, with overlapping.? How to measure performance?. What are the labels?.
	Answer: Patches not needed. Labels are 1 in lesion, 0 in no lesion.



## May 25 - May 26
### Activities
* Installed in CTS 5
* Read cs231n Stanford Convnet course (cs231n.github.io).

### To do
* End writing Background
	* ConvNet
	* ConvNet for Breast Cancer
	* Practical Deep Learning
	* Database specifics
* Update LaTex template to thesis template
* Choose software (probably Caffe)
* Select which forms of preprocessing to try

### Questions
1. Should I preserve a test set just for the final step (in December) or is it ok to use all data for the preprocessing choosing and then all data for the small vs big and all that?. 
	Answer: Separate a test set right at the beginning. Treat preprocessing as a hyperparameter to fit. For transfer leraning and big vs small you can use the entire dataset but shuffle the test set to be different. 

2. Validation or 5-fold crossvalidation?
	Answer: Validation. If validation set is too small, then 5-fold.

3. When checking for different preprocessings, fit all hyperparameters or only a subset or none at all?
	Answer:	Fit learning rate and regularization. All other hyperparameters would be set to standard (including the network architecture). 

4. mxn o nxd for the name of dimensions?
	Answer: mxn. m examples of n dimensions.

5. Is it a binary classification(cancer/no cancer) or 3 classes (micro/mass/nothing) or something else(detection).
	Answer: Image segmentation. Lesion vs. background.

6. Which forms of image enhancement should I use?. No preprocessing or global contrast stretching?
