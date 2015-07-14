# Some figures to see what's the best way to resize the mammograms. Using scipy.misc.imresize with old PIL version. Results are probably not valid.

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

mammogram = scipy.misc.imread("mammogram.jpg")
>>> np.shape(mammogram)
(3328, 2560)
>>> np.mean(mammogram)
65.439185744065512
>>> mammogramB = mammogram.copy()
>>> mammogramB[mammogramB <= 65] = 0
>>> mammogramB[mammogramB > 65] -=65
>>> scipy.misc.imsave("mammogramB.png",mammogramB)

# Just to check both produce the same output (remember imshow normalizes automatically). It does.
>>> mammogramB2 = mammogram.copy()
>>> mammogramB2[mammogramB2 <= 65] = 65
>>> plt.subplot(1,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fe2c61d05c0>
>>> plt.imshow(mammogramB)
<matplotlib.image.AxesImage object at 0x7fe2c929bcc0>
>>> plt.subplot(1,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fe2c453ba90>
>>> plt.imshow(mammogramB2)
<matplotlib.image.AxesImage object at 0x7fe2c4596d30>
>>> plt.gray()
>>> plt.show()

# do the same with a patch
>>> patch = scipy.misc.imread("patch.jpg")
>>> np.shape(patch)
(404, 404, 3)
>>> patch= patch[:,:,0]
>>> np.mean(patch)
137.12805117145379
>>> patchB = patch.copy()
>>> patchB[patchB <= 137] = 137
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fe2bc1144a8>
>>> plt.imshow(patch, vmin= 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fe2d2d93f98>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fe2c450a320>
>>> plt.imshow(patch)
<matplotlib.image.AxesImage object at 0x7fe2c450a390>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fe2c457f240>
>>> plt.imshow(patchB)
<matplotlib.image.AxesImage object at 0x7fe2c457f390>
>>> plt.show()

# what happens with an all black image?. It remains black
>>> blackImage = np.zeros((200,200))
>>> np.shape(blackImage)
(200, 200)
>>> plt.imshow(blackImage)
<matplotlib.image.AxesImage object at 0x7fe2bc141978>
>>> plt.show()

# Resizing the image. Original size is 2560 x 3328 pixels (width x height) at 0.05 mm pixel size (I'm actually  not sure about this, but is probably around there). To resize to pixel size of 0.32 mm it would be 400 x 520 pixels image exactly. For pixel size 0.16 mm the size is double that.
>>> smallMammogram = scipy.misc.imresize(mammogram, (520,400))
>>> scipy.misc.imsave("smallMammogram.png", smallMammogram)
>>> smallPatch = scipy.misc.imresize(patch, (63,63))
>>> scipy.misc.imsave("smallPatch.png", smallPatch)

# Try all interpolations. They all look the same, I like cubic.
smallPatch1 = scipy.misc.imresize(patch, (63,63), interp = "nearest")
smallPatch2 = scipy.misc.imresize(patch, (63,63), interp = "bilinear")
smallPatch3 = scipy.misc.imresize(patch, (63,63), interp = "cubic")
smallPatch4 = scipy.misc.imresize(patch, (63,63), interp = "bicubic")

plt.subplot(2,2,1)
plt.imshow(smallPatch1)
plt.title("Nearest")
plt.subplot(2,2,2)
plt.imshow(smallPatch2)
plt.title("Bilinear")
plt.subplot(2,2,3)
plt.imshow(smallPatch3)
plt.title("Cubic")
plt.subplot(2,2,4)
plt.imshow(smallPatch4)
plt.title("Bicubic")
plt.show()

plt.subplot(2,2,1)
plt.imshow(smallPatch1, interpolation = "none")
plt.title("Nearest")
plt.subplot(2,2,2)
plt.imshow(smallPatch2, interpolation = "none")
plt.title("Bilinear")
plt.subplot(2,2,3)
plt.imshow(smallPatch3, interpolation = "none")
plt.title("Cubic")
plt.subplot(2,2,4)
plt.imshow(smallPatch4, interpolation = "none")
plt.title("Bicubic")
plt.show()

smallPatch1B = smallPatch1.copy()
smallPatch1B[smallPatch1B <= 138] = 138
smallPatch2B = smallPatch2.copy()
smallPatch2B[smallPatch2B <= 138] = 138
smallPatch3B = smallPatch3.copy()
smallPatch3B[smallPatch3B <= 138] = 138
smallPatch4B = smallPatch4.copy()
smallPatch4B[smallPatch4B <= 138] = 138

plt.subplot(2,2,1)
plt.imshow(smallPatch1B, interpolation = "none")
plt.title("Nearest -> BR")
plt.subplot(2,2,2)
plt.imshow(smallPatch2B, interpolation = "none")
plt.title("Bilinear -> BR")
plt.subplot(2,2,3)
plt.imshow(smallPatch3B, interpolation = "none")
plt.title("Cubic -> BR")
plt.subplot(2,2,4)
plt.imshow(smallPatch4B, interpolation = "none")
plt.title("Bicubic -> BR")
plt.show()

plt.subplot(2,2,1)
plt.imshow(smallPatch1B)
plt.title("Nearest -> BR")
plt.subplot(2,2,2)
plt.imshow(smallPatch2B)
plt.title("Bilinear -> BR")
plt.subplot(2,2,3)
plt.imshow(smallPatch3B)
plt.title("Cubic -> BR")
plt.subplot(2,2,4)
plt.imshow(smallPatch4B)
plt.title("Bicubic -> BR")
plt.show()

# Different views for the cubic interpolation
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fab09d05400>
>>> plt.imshow(smallPatch3)
<matplotlib.image.AxesImage object at 0x7fab155e3e48>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fab0ad8fe48>
>>> plt.imshow(smallPatch3B)
<matplotlib.image.AxesImage object at 0x7fab0ad823c8>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fab0af6b4a8>
>>> plt.imshow(smallPatch3, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fab0ad826a0>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fab0af6b160>
>>> plt.imshow(smallPatch3B, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fab09cd9d68>
>>> plt.show()

# Want to see how much resolution is lost from the reduction, should be a lot because 36 pixels get reduced to a single one. Wow it's bad!
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fab09e31780>
>>> plt.imshow(patchB)
<matplotlib.image.AxesImage object at 0x7fab0ad6fcf8>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fab09d68e80>
>>> plt.imshow(smallPatch3B)
<matplotlib.image.AxesImage object at 0x7fab09e2d518>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fab09d05c88>
>>> plt.imshow(patchB, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fab0af015c0>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fab0ae94da0>
>>> plt.imshow(smallPatch3B, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fab09c04160>
>>> plt.show()
