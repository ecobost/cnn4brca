import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
plt.gray()

m = scipy.misc.imread("img_174_233_1_LCC.tif")
m.shape
# (3045, 1332, 4)
m = m[:,:,1]m.shape
#(3045, 1332)
im = Image.fromarray(m)
im64 = im.resize((298,682), Image.LANCZOS)
im96 = im.resize((448,1023), Image.LANCZOS)
im128 = im.resize((597,1364), Image.LANCZOS)
scipy.misc.imsave("im64.png",im64)
scipy.misc.imsave("im96.png",im96)
scipy.misc.imsave("im128.png",im128)
plt.imshow(np.array(im), interpolation = "none")
plt.subplot(1,4,1)
plt.imshow(np.array(im), interpolation = "none")
plt.subplot(1,4,2)
plt.imshow(np.array(im128), interpolation = "none")
plt.subplot(1,4,3)
plt.imshow(np.array(im96), interpolation = "none")
plt.subplot(1,4,4)
plt.imshow(np.array(im64), interpolation = "none")
plt.show()

# Cut a 286x286 patch and resize
mp = m[1100:1386,450:736]
imp = Image.fromarray(mp)
imp64 = imp.resize((64,64), Image.LANCZOS)
imp96 = imp.resize((96,96), Image.LANCZOS)
imp128 = imp.resize((128,128), Image.LANCZOS)
scipy.misc.imsave("imp64.png",imp64)
scipy.misc.imsave("imp96.png",imp96)
scipy.misc.imsave("imp128.png",imp128)
plt.imshow(np.array(imp), interpolation = "none")
plt.subplot(1,4,1)
plt.imshow(np.array(imp), interpolation = "none")
plt.subplot(1,4,2)
plt.imshow(np.array(imp128), interpolation = "none")
plt.subplot(1,4,3)
plt.imshow(np.array(imp96), interpolation = "none")
plt.subplot(1,4,4)
plt.imshow(np.array(imp64), interpolation = "none")
plt.show()



#!!!plt and scipy.misc.imsave cannot handel mode '1' images use IMage.open() im.save() and things like that

# downsampling the mask/labels
# I need to convert to uint8, otherwise it may not work
im = Image.open("img_174_233_1_LCC_mask.png")
im2 = im.convert('L')
imLarge = im2.resize((1304, 1600), Image.LANCZOS)
imSmall = im2.resize((82, 100), Image.LANCZOS)
plt.subplot(1,2,1)
plt.imshow(np.array(imLarge), interpolation = "none")
plt.subplot(1,2,2)
plt.imshow(np.array(imSmall), interpolation = "none")
plt.show()

#Make it a mask
sm = np.array(imSmall)
sm[sm>10] = 255
sm[sm<=10] = 0
plt.imshow(sm, interpolation = "none")
plt.show()


#Enlarge with bilinear to see what happens
imSmallBigger = imSmall.resize((1304, 1600),Image.BILINEAR)
plt.imshow(np.array(imSmallBigger), interpolation = "none")
plt.show()

#Trying to resize directly from binary to binary. It works better, uses Nearest neighbor
# plt or scipy.misc.imsave cannot handle it
imSmallBinary = im.resize((82, 100)) # If mode '1', Nearest neighbor is used by default
imSmallBinary
#<PIL.Image.Image image mode=1 size=100x82 at 0x7FE829ABA080>
imSmallBinary.save("i.png")
#For plt to handle it (and to assign breast area) convert to uint8
imSmallBinary2 = imSmallBinary.convert('L')
imSmallBinary2.save("i2.png")

# Generating labels with breast area
mam = scipy.misc.imread("img_174_233_1_LCC_2.tif")
mask = np.array(im2)
label = mask.copy()
condition = np.logical_and(mask == 0, mam > 0)
label[condition] = 127
imLabel = Image.fromarray(label)
imLabel
#<PIL.Image.Image image mode=L size=3328x4084 at 0x7FE825053A90>
imLabel.save("label.png")

#Resizing the entire label (use NEAREST to make sure everything is a proper label)
imLabelSmall = imLabel.resize((82, 100), Image.NEAREST)
imLabelLarge = imLabel.resize((1304, 1600), Image.NEAREST)
plt.imshow(np.array(imLabelSmall), interpolation = "none")
plt.show()
imLabelSmall.save("smallLabel.png")








