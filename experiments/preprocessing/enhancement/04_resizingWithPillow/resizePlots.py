# Summary: 
#	PILLOW interps looks better than PIL
#	NEAREST interpolation is pixelated, all other #look blurry (rather than pixelated)
#	BICUBIC and LANCZOS look pretty much the same. 
#	Enhancement first and resizing later or resizing first and enhancement later is not important, pretty much same results.
# How to install http://askubuntu.com/questions/427358/install-pillow-for-python-3 

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

patch = scipy.misc.imread("patch.jpg")
patch.shape
# (404, 404, 3)
patch= patch[:,:,0]
patch.mean()
#137.12805117145379im
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

# Test the BICUBIC and LANCZOS of the new PILLOW (supposedly better). ANTIALIAS and LANCZOS are exactly the same, bicubic is not pixelated anymore, looks like LANCZOS
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

# Test all methods with enhancement first (patchB) and then reduction and with reduction first and enhancement later.
patch = scipy.misc.imread("breastMicrocalcification.jpg")
patch= patch[:,:,0]
import normalize
patchB = normalize.adjustContrast(patch)
imB = Image.fromarray(patchB)
patchB2 = imB.resize((105,105), Image.BICUBIC)
patchB3 = imB.resize((105,105), Image.LANCZOS)

im = Image.fromarray(patch)
patch2 = np.array( im.resize((105,105), Image.BICUBIC) )
patch2 = normalize.adjustContrast(patch2)
patch3 = np.array( im.resize((105,105), Image.LANCZOS) )
patch3 = normalize.adjustContrast(patch3)

plt.subplot(2,2,1)
plt.title("BR -> Bicubic")
plt.imshow(patchB2, interpolation = "none", vmin = 0, vmax = 255)
plt.subplot(2,2,2)
plt.title("BR -> Lanczos")
plt.imshow(patchB3, interpolation = "none", vmin = 0, vmax = 255)
plt.subplot(2,2,3)
plt.title("Bicubic -> BR")
plt.imshow(patchB2, interpolation = "none", vmin = 0, vmax = 255)
plt.subplot(2,2,4)
plt.title("Lanczos -> BR")
plt.imshow(patchB3, interpolation = "none", vmin = 0, vmax = 255)
plt.show()
