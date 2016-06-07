# Activity Log
Written by: Erick Cobos T. (a01184587@itesm.mx)

Log with activities and questions arising during every week

## Jun 2 - Jun 8
### Activities
* Write Experiment 3 in Ch 3
* Write code for Experiment 3

### To do
* Run Experiment 3 Hyperparameter search
* Run Experiment 3
* Ask for institutional email


## May 19 - May 25
### Activities
* Train best Experiment 2 model
* Write results for Experiment 2 in Chapter 4

### To do
* Write code for model 3
* Ask for institutional email

### Questions
1. Should I try using a Resnet-like model with dilated convolutions?
	Answer: Sure :)
2. Should I try a simpler model with less parameters (maybe in exchange for more memory usage)?


## May 12 - May 18
### Activities
* Write code to test models in test set
* Run hyperparameter search for Experiment 2
* Write hyperparameter search result for Exp 2

### To do
* Train best Experiment 2 model
* Write results for Experiment 2 in Chapter 4
* Ask for institutional email


## May 5 - May 11
### Activities
* Write Chapter 3 in thesis
* Write results from Experiment 1 in thesis.

### To do
* Change loss function for experiment 2
* Run Experiment 2 hyperparameter search
* Ask for institutional email


## Apr 29 - May 4
### Activities
* Run refined hyperparameter search
* Train best network

### To do
* Write hyperparameter results in thesis
* Write results from the best network
* Ask for institutional email


## Apr 25 - Apr 28
### Activities
* Run hyperparameter search
* Get hyperparameter figures

### To do
* Refine hyperparameter search
* Write Model and Results
* Train best network
* Ask for institutional email


## Apr 14 - Apr 22
### Activities
* Install CUDA and Tensorflow 0.8.0 in computers in A3-401
* Modify model to run inside 2GB GPUs
* Do the slides for the Research congress
* Expose in research congress

### To do
* Run hyperparamter search
* Write Section 3.2(Model) and 3.3(Training)
* Ask for institutional email


## Apr 07 - Apr 13
### Activities
* Install CUDA and Tensorflow in computers in A3-401
* Modify model to run in TF 0.8.0

### To do
* Install Tensorflow in computers at A3-301
* Run experiments
* Write Section 3.2(Model) and 3.3(Training)
* Ask for institutional email


## Mar 31 - Apr 06
### Activities
* Writing Tensorflow implementation
* Run sanity checks and initial tests

### To do
* Install Tensorflow in computers at A3-401
* Run experiments
* Write Section 3.2(Model) and 3.3(Training)
* Ask for institutional email


## Mar 17 - Mar 30
### Activities
* Spoke with Dr. Parra
* Writing Tensorflow implementation

### To do
* Write network in Tensorflow
* Install Tensorflow in computers at A3-401
* Run experiments
* Write Section 3.2(Model) and 3.3(Training)
* Ask for institutional email


## Mar 10 - Mar 16
### Activities
* Spoke with Dr. Garrido
* Writing Tensorflow implementation

### To do
* Write network in Tensorflow
* Ask for a computer in A3-401 (Dr. Parra) or the cluster of CPUs (Dr. Nolazco)
* Write chapter 3 (Solution Model)
* Ask for institutional email


## Mar 04 - Mar 09
### Activities
* Reading Tensorflow tutorials

### To do
* Write network in Tensorflow
* Ask for a computer in A3-401 (Dr. Parra) or the cluster of CPUs (Dr. Nolazco)
* Write chapter 3 (Solution Model)
* Ask for institutional email


## Feb 25 - Mar 03
### Activities
* Preprocessing database
* Writing Section 3.2
* Writing Section 3.3
* Watching cs231n lectures/ updating knowledge
* Installing Tensorflow (Laptop/CTS)
* Reading Tensorflow tutorials

### To do
* Read Tensorflow tutorials
* Write network in Tensorflow
* Ask for a computer in A3-401 or somewhere else (maybe ask Dr. Garrido)
* Write chapter 3 (Solution Model)
* Ask for institutional email

### Questions
1. Should I add code as appendix (prepareDB.py or tensorflow.py) or link to the github project or not do any?
	Answer: Put a link to Github


## Feb 18 - Feb 24
### Activities
* Writing Section 2.6.2, adding to 2.5
* Taking more desing decisions
* Writing Section 3.2

### To do
* Ready database
* Write chapter 3 (Solution Model)
* Install TensorFlow (CTS/Laptop)
* Write network in Tensorflow/Keras
* Ask for a computer in A3-401 or somewhere else (maybe ask Dr. Garrido)
* Ask for institutional email

### Questions
1. Should I leave on the database part in Breast cancer or better just delete it and degrade mammograms and maybe put BCDR info in Model/Data set/Database?
	Answer: Leave it there.
2. Should I try to fit a simple model and a more advanced model (ADAM, batchnorm, leaky relus) or should I go directly for the best model?
	Answer: Best model.


## Feb 12 - Feb 17
### Activities
* Writing section 2.6 and 2.7
* Writing Section 3.1

### To do
* Write chapter 3 (Solution Model)
* Ready database
* Install TensorFlow (CTS/Laptop)
* Write network in Tensorflow/Keras
* Ask for a computer in A3-401 or somewhere else (maybe ask Dr. Garrido)
* Ask for institutional email

### Questions
1. Is IOU fine for unbalanced data sets?
	Answer: For model selection IOU is going to try to maximize the intersection and minimize the union as the union is waaay bigger (because objects are small), it will probably try to minimize the union more prediciting less positive labels and it may lose sensitivity (for the sake of specificity). Not sure about this, though, seems like F-1 is gonna do the same.
2. Should I leave Section 2.7 citations as "[23] trained ..." or write "Ge et al. trained"?
	Answer: Say names
3. For background, should I cite all articles where a netwrok appear, if for example they reported something twice.
	Answer: No
4. In the solution model, should I write all alternatives, say what I chose, and explain why I choose the one i chose or just say what I chose?
	Answer: Just what you chose and justify why. Maybe an alternative but only slightly.


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
2. Which evaluation metric should I use? Accuracy, F1-score, PRAUC, ROC, IOU, Dice?
	Answer: IOU. F1-score in second place.
3. Should I cite Agarwal2015 (unpublished Stanford report)?
	Answer: No.

---------------------------------------------------------------------------
---------------------------------------------------------------------------

## Jul 2 - Jul 8
### Activities
* Write emails looking for more digital mammograms
* Document decisions on how to crop big images.

### To do
* Ask for institutional email.


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
	Answer: Segmentation (mass(benign or malign) vs non-mass)
2. Should I use a single network with multiple outputs to classify every kind of lession?.
	Answer: No.
3. Should I use data augmentation only on the minority classes (lessions)?
	Answer: Use everywhere. No oversampling


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
