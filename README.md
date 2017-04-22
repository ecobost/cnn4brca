# cnn4brca
Using Convolutional Neural Networks (CNN) for Semantic Segmentation of Breast Cancer Lesions (BRCA). Master's thesis documents. Bibliography, experiments and reports.

Most articles in the Bibliography folder were obtained directly from the authors or via agreements with my home institution. Please consider any copyright infringement before using them.

##### Contact info:
Erick Cobos Tandazo<br>
a01184587@itesm.mx

## Usage
### Data set
1. You can obtain the BCDR database [online](http://bcdr.inegi.up.pt/) ([Moura et al.](http://dx.doi.org/10.1007/s11548-013-0838-2)). I used the BCDR-DO1 data set, this one has around 70 patients(~300 digital mammograms) with breast masses and their lesion outlines. [fileOrganization](database_info/file_Organization) has some info on how is this images ordered.

2. To obtain the masks (from the outlines provided in the database) you can use [createMasks.m](database_info/createMask/createMask.m). This reads the mammogram info from a couple of files provided in the database: [sample bcdr_d01_img.csv](database_info/createMask/bcdr_d01_img.csv) and [sample bcdr_d01_outlines.csv](database_info/createMask/bcdr_d01_outlines.csv)
   Output should look like this:
   <img src="database_info/createMask/img_20_30_1_RCC.png" width="250"/> <img src="database_info/createMask/img_20_30_1_RCC_mask.png" width="250"/> 

3. Use [prepareDB](code/prepareDB.py) to enhance the contrast of the mammograms and downsample them to have a manageable size (2cmx2cm in the mammogram in 128x128).
   Output looks like this:
   <img src="docs/report/plots/mammogram_resized.png" width="250" align='center'>

4. Finally you would need to divide the dataset into training, validation and test patients. You would need to produce a .csv with image and label filenames as [this](code/example.csv) for each set.

### Training
1. You would need to [install Tensorflow](https://www.tensorflow.org/install/)
2. Run [train](code/train.py) or [train_with_val_split](code/train_with_val_split.py) to train networks. These train the network defined in [model_v3](code/model_v3.py), a fully convolutional network with 10 layers (900K parameters) that uses dillated convolution and is modelled in a ResNet network. Training is done image by image (no batch, but cost is computed in every pixel of the thousand of pixels) and uses dropout among other things
    Note: Code was written for tensorflow 1.11.0 so it would need to be modified to make work in tf1.0

### Evaluation
1. You can use [compute_metrics](code/compute_metrics.py) or [compute_FROC](code/compute_FROC.py) to compute evaluation metrics or the FROC curve.

You are invited to check the code for more details, I tried to document it nicely.
