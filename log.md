# Activity Log
Written by: Erick Cobos T. (a01184587@itesm.mx)

Log with activities and questions arising during every week

## May 27- Jun 2
### Activities
*
### To do
*
### Questions
1.

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

	Answer: Separate a test set right at the beginning. Treat preprocessing as a hyperparameter to fit. For transfer leraning and big vs small you cna use the entire dataset but shuffle the test set to be different. 

2. Validation or 5-fold crossvalidation?

	Answer: Validation. If validation ste is small, then 5-fold.

3. When checking for different preprocessings, fit all hyperparameters or only a subset or none at all?

	Answer:	Fit learning rate and regularization. All other hyperparams would be set to standard (including the network architecture). 

4. mxn o nxd for the name of dimensions?
	
	Answer: mxn. m examples of n dimensions.

5. Is it a binary classification(cancer/no cancer) or 3 classes (micro/mass/nothing) or something else(detection)?
6. Which forms of preprocessing should I use?. 
