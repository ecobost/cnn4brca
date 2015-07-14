# Summary: 
#	PILLOW interps looks better than PIL
#	NEAREST interpolation is pixelated, all other #look blurry (rather than pixelated)
#	BICUBIC and LANCZOS look pretty much the same. 
NOT YET!!!!#	Enhancement first and resizing later or resizing first and enhancement later is not important, pretty much same results.

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

patch = scipy.misc.imread("patch.jpg")
patch.shape
# (404, 404, 3)
patch= patch[:,:,0]
patch.mean()
#137.12805117145379
patchB = patch.copy()
patchB[patchB <= 137] = 137

# Test the BICUBIC vs "ANTIALIASING" PIL interpolations (old PIL, not PILLOW) 
from PIL import Image 
im = Image.fromarray(patch)
im.size
# (404, 404)
smallImage1 = im.resize((63,63), Image.BICUBIC)
smallImage2 = im.resize((63,63), Image.ANTIALIAS)
smallImage1 = np.array(smallImage1)
smallImage2 = np.array(smallImage2)
smallImage1.mean()
# 137.67573696145124
np.mean(smallImage2)
# 137.13454270597128
smallImage1B = smallImage1.copy()
smallImage2B = smallImage2.copy()
smallImage1B[ smallImage1B <= 138] = 138
smallImage2B[ smallImage2B <= 137] = 137

plt.subplot(1,3,1)
plt.imshow(patchB, interpolation = "none")
plt.title("Full size")
plt.subplot(1,3,2)
plt.imshow(smallImage1B, interpolation = "none")
plt.title("Bicubic")
plt.subplot(1,3,3)
plt.title("Antialiasing")
plt.imshow(smallImage2B, interpolation = "none")
plt.show()

# Test the BCUBIC and LANCZOS of the new PILLOW (supposedly better). ANTIALIAS and LANCZOS are exactly the same, bicubic is not pixelated anymore, looks like LANCZOS
im = Image.fromarray(patch)
im.size
# (404, 404)
smallImage1 = im.resize((63,63), Image.BICUBIC)
smallImage2 = im.resize((63,63), Image.LANCZOS)
smallImage1 = np.array(smallImage1)
smallImage2 = np.array(smallImage2)
smallImage1.mean()
# 137.14184933232553
smallImage2.mean()
# 137.13454270597128
smallImage1B = smallImage1.copy()
smallImage2B = smallImage2.copy()
smallImage1B[ smallImage1B <= 137] = 137
smallImage2B[ smallImage2B <= 137] = 137

plt.subplot(1,3,1)
plt.imshow(patchB, interpolation = "none")
plt.title("Full size")
plt.subplot(1,3,2)
plt.imshow(smallImage1B, interpolation = "none")
plt.title("Bicubic")
plt.subplot(1,3,3)
plt.title("Lanczos")
plt.imshow(smallImage2B, interpolation = "none")
plt.show()

# Test the other methods of reduction
smallPatch1 = np.array( im.resize((63,63), Image.NEAREST) )
smallPatch2 = np.array( im.resize((63,63), Image.BILINEAR) )
smallPatch3 = np.array( im.resize((63,63), Image.BICUBIC) )
smallPatch4 = np.array( im.resize((63,63), Image.LANCZOS) ) 
smallPatch1.mean()
#138.06676744771983
smallPatch2.mean()
#137.1310153691106
smallPatch3.mean()
#137.14184933232553
smallPatch4.mean()
#137.13454270597128


smallPatch1B = smallPatch1.copy()
smallPatch1B[smallPatch1B <= 138] = 138
smallPatch2B = smallPatch2.copy()
smallPatch2B[smallPatch2B <= 137] = 137
smallPatch3B = smallPatch3.copy()
smallPatch3B[smallPatch3B <= 137] = 137
smallPatch4B = smallPatch4.copy()
smallPatch4B[smallPatch4B <= 137] = 137

plt.subplot(2,2,1)
plt.imshow(smallPatch1B, interpolation = "none")
plt.title("Nearest -> BR")
plt.subplot(2,2,2)
plt.imshow(smallPatch2B, interpolation = "none")
plt.title("Bilinear -> BR")
plt.subplot(2,2,3)
plt.imshow(smallPatch3B, interpolation = "none")
plt.title("Bicubic -> BR")
plt.subplot(2,2,4)
plt.imshow(smallPatch4B, interpolation = "none")
plt.title("Lanczos -> BR")
plt.show()

# Test all methods with enhancement first (patchB) and then reduction. Same thing
!!!!!!!!!!!!!! PatchB needs to be normalized first
im = Image.fromarray(patchB)
smallPatch1 = np.array( im.resize((63,63), Image.NEAREST) )
smallPatch2 = np.array( im.resize((63,63), Image.BILINEAR) )
smallPatch3 = np.array( im.resize((63,63), Image.BICUBIC) )
smallPatch4 = np.array( im.resize((63,63), Image.LANCZOS) ) 

plt.subplot(2,2,1)
plt.imshow(smallPatch1B, interpolation = "none")
plt.title("BR -> Nearest")
plt.subplot(2,2,2)
plt.imshow(smallPatch2B, interpolation = "none")
plt.title("BR -> Bilinear")
plt.subplot(2,2,3)
plt.imshow(smallPatch3B, interpolation = "none")
plt.title("BR -> Bicubic")
plt.subplot(2,2,4)
plt.imshow(smallPatch4B, interpolation = "none")
plt.title("BR -> Lanczos")
plt.show()

# read tutorial, find normalization/contrast stretching,
# Compare normal vs bicubic vs lanczos (new version). draw 6 images, first row for reduction then enhanecement and the second one for enhancement then reduction. 
# Stride is harder if I reduce later because I have to stride on images of different sizes so the 6 pixels will be different. All about what's better.
# What about the contours in the smaller picture, how do i resize the irregular conoturs?.
# Having the big patches may be easier. The stride may not be exactly 6 pixels, the size of the spatial resolution does have to be exactly 6 tohgh




>>> smallPatchB = smallPatch.copy()
>>> smallPatchB[ smallPatchB <= 137] = 137
>>> plt.subplot(1,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fe2afdc1710>
>>> plt.imshow(patchB)
<matplotlib.image.AxesImage object at 0x7fe2bc11a780>
>>> plt.subplot(1,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fe2c450a860>
>>> plt.imshow(smallPatchB)
<matplotlib.image.AxesImage object at 0x7fe2c0309940>
>>> plt.show()

tuesday 6:30 hours
