# Summary: Use background reduction plus normalization or simple normalization.

import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

# plt.imshow() does constrast strecthing with range equal to the range in the image or vmin, vmax if passed to it.
# According to these (http://www.mathworks.com/help/images/adjusting-image-contrast-using-the-adjust-contrast-tool.html#brc1_4s-1) constrast stretching stretches the contrast between two specified points. " pixel values below a specified value are displayed as black, pixel values above a specified value are displayed as white, and pixel values in between these two values are displayed as shades of gray."

>>> mass = scipy.misc.imread("breastMass.jpg")
>>> np.shape(mass)
(430, 429, 3)
>>> mass = mass[:,:,0]
>>> np.mean(mass)
172.95355342332087
>>> massB = mass.copy()
>>> massB[massB <= 173] = 0
>>> massB[massB > 173] -= 173
>>> massB2 = mass.copy()
>>> massB2[massB2 <= 86] = 0
>>> massB2[massB2 > 86] -= 86
>>> massB3 = mass.copy()
>>> massB3[massB3 <= 173] = 0
>>> scipy.misc.imsave("massB.png",massB)
>>> scipy.misc.imsave("massB2.png",massB2)
>>> scipy.misc.imsave("massB3.png",massB3)
>>> plt.gray()
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f19e668>
>>> plt.title("Normal mass")
<matplotlib.text.Text object at 0x7fc47f7876d8>
>>> plt.imshow(mass)
<matplotlib.image.AxesImage object at 0x7fc47f799c50>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f799c88>
>>> plt.title("Mean Background reduction")
<matplotlib.text.Text object at 0x7fc47f74a208>
>>> plt.imshow(massB)
<matplotlib.image.AxesImage object at 0x7fc47f767748>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f767780>
>>> plt.title("Half Mean Background reduction")
<matplotlib.text.Text object at 0x7fc47f702898>
>>> plt.imshow(massB2)
<matplotlib.image.AxesImage object at 0x7fc47f71f550>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47f71f588>
>>> plt.title("Background reduction wo substract.")
<matplotlib.text.Text object at 0x7fc47f0ceac8>
>>> plt.imshow(massB3)
<matplotlib.image.AxesImage object at 0x7fc47f0f0048>
>>> plt.show()

# No normalization.
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f0444e0>
>>> plt.title("Normal mass")
<matplotlib.text.Text object at 0x7fc47f170be0>
>>> plt.imshow(mass,vmin = 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f12dd30>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f18ca58>
>>> plt.title("Mean Background reduction")
<matplotlib.text.Text object at 0x7fc47f021fd0>
>>> plt.imshow(massB,vmin = 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f029550>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f029ac8>
>>> plt.title("Half Mean Background reduction")
<matplotlib.text.Text object at 0x7fc47f8bb5c0>
>>> plt.imshow(massB2,vmin = 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f8d7278>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47f8d77f0>
>>> plt.title("Background reduction wo substract.")
<matplotlib.text.Text object at 0x7fc47f8857f0>
>>> plt.imshow(massB3,vmin = 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f8a1d30>
>>> plt.show()


# Microcalcifications
>>> mc = scipy.misc.imread("breastMicrocalcification.jpg")
>>> np.shape(mc)
(337, 337, 3)
>>> mc = mc[:,:,0]
>>> np.mean(mc)
171.04156063714569
>>> mcB= mc.copy()
>>> mcB[mcB <= 171] = 0
>>> mcB[mcB > 171] -= 171
>>> mcB2 = mc.copy()
>>> mcB2[mcB2 <= 85] = 0
>>> mcB2[mcB2>85] -= 85
>>> mcB3 = mc.copy()
>>> mcB3[mcB3 <= 171] = 0
>>> scipy.misc.imsave("mcB.png",mcB)
>>> scipy.misc.imsave("mcB2.png",mcB2)
>>> scipy.misc.imsave("mcB3.png",mcB3)
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f844fd0>
>>> plt.title("Normal Microcalcifications")
<matplotlib.text.Text object at 0x7fc47f08ac18>
>>> plt.imshow(mc)
<matplotlib.image.AxesImage object at 0x7fc47f7a8d68>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f7a8fd0>
>>> plt.title("Mean Background Reduction")
<matplotlib.text.Text object at 0x7fc47f73fd30>
>>> plt.imshow(mcB)
<matplotlib.image.AxesImage object at 0x7fc47f0af668>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f0aff28>
>>> plt.title("Half Mean Background Reduction")
<matplotlib.text.Text object at 0x7fc47f0b4ac8>
>>> plt.imshow(mcB2)
<matplotlib.image.AxesImage object at 0x7fc47f0d4c18>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47f0d4b70>
>>> plt.title("Background Reduction wo substract")
<matplotlib.text.Text object at 0x7fc47f03bd30>
>>> plt.imshow(mcB3)
<matplotlib.image.AxesImage object at 0x7fc47f047780>
>>> plt.show()

# No normalization
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f8da6d8>
>>> plt.title("Normal Microcalcifications")
<matplotlib.text.Text object at 0x7fc47f6f7cf8>
>>> plt.imshow(mc,vmin=0,vmax=255)
<matplotlib.image.AxesImage object at 0x7fc47f7677f0>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f7677b8>
>>> plt.title("Mean Background Reduction")
<matplotlib.text.Text object at 0x7fc47f0cb5f8>
>>> plt.imshow(mcB,vmin=0,vmax=255)
<matplotlib.image.AxesImage object at 0x7fc47f767e48>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f7a8208>
>>> plt.title("Half Mean Background Reduction")
<matplotlib.text.Text object at 0x7fc47f139198>
>>> plt.imshow(mcB2,vmin=0,vmax=255)
<matplotlib.image.AxesImage object at 0x7fc47f7a82b0>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47f73f630>
>>> plt.title("Background Reduction wo substract")
<matplotlib.text.Text object at 0x7fc47f7a27b8>
>>> plt.imshow(mcB3,vmin=0,vmax=255)
<matplotlib.image.AxesImage object at 0x7fc47f1160f0>
>>> plt.show()

# Histogram Equalization
# Code taken from: http://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy
>>> imhist,bins = np.histogram(mc.flatten(), 256 ,normed=True)
>>> cdf = imhist.cumsum()
>>> cdf = 255 * cdf / cdf[-1] #normalize
>>> im2 = np.interp(mc.flatten(),bins[:-1],cdf)
>>> mcH = im2.reshape(mc.shape)
>>> scipy.misc.imsave("mcH.png",mcH)
>>> imhist,bins = np.histogram(mass.flatten(), 256 ,normed=True)
>>> cdf = imhist.cumsum()
>>> cdf = 255 * cdf / cdf[-1] #normalize
>>> im2 = np.interp(mass.flatten(),bins[:-1],cdf)
>>> massH = im2.reshape(mass.shape)
>>> scipy.misc.imsave("massH.png", massH)

# Normal vs Normalized/Constrast Stretching vs Contrast Stretching plus Background Reduction vs Histogram Equalization
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f0204e0>
>>> plt.imshow(mass, vmin= 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f0eafd0>
>>> plt.title("No preprocessing")
<matplotlib.text.Text object at 0x7fc47f0f9160>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f8aeb00>
>>> plt.title("Normalized image")
<matplotlib.text.Text object at 0x7fc47f0b4048>
>>> plt.imshow(mass)
<matplotlib.image.AxesImage object at 0x7fc47f850c88>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f850198>
>>> plt.title("Background Reduction + Norm")
<matplotlib.text.Text object at 0x7fc47f081eb8>
>>> plt.imshow(massB)
<matplotlib.image.AxesImage object at 0x7fc47f767630>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47f767ba8>
>>> plt.title("Histogram Equalization")
<matplotlib.text.Text object at 0x7fc47f110278>
>>> plt.imshow(massH, vmin= 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f83be48>
>>> plt.show()

>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f73f9b0>
>>> plt.imshow(mc, vmin= 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f7f69b0>
>>> plt.title("No preprocessing")
<matplotlib.text.Text object at 0x7fc47f006ef0>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f7074e0>
>>> plt.imshow(mc)
<matplotlib.image.AxesImage object at 0x7fc47f707cc0>
>>> plt.title("Normalized image")
<matplotlib.text.Text object at 0x7fc47f8e2828>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f8b6828>
>>> plt.title("Background Reduction + Norm")
<matplotlib.text.Text object at 0x7fc47f8c9550>
>>> plt.imshow(mcB)
<matplotlib.image.AxesImage object at 0x7fc47ee3bc88>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47ee3b908>
>>> plt.title("Histogram Equalization")
<matplotlib.text.Text object at 0x7fc47f7f7d68>
>>> plt.imshow(mcH, vmin= 0, vmax = 255)
<matplotlib.image.AxesImage object at 0x7fc47f7eef28>
>>> plt.show()

# No interpolation
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7fc47f73f9b0>
>>> plt.imshow(mc, vmin= 0, vmax = 255, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fc47f7f69b0>
>>> plt.title("No preprocessing")
<matplotlib.text.Text object at 0x7fc47f006ef0>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7fc47f7074e0>
>>> plt.imshow(mc, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fc47f707cc0>
>>> plt.title("Normalized image")
<matplotlib.text.Text object at 0x7fc47f8e2828>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7fc47f8b6828>
>>> plt.title("Background Reduction + Norm")
<matplotlib.text.Text object at 0x7fc47f8c9550>
>>> plt.imshow(mcB, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fc47ee3bc88>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7fc47ee3b908>
>>> plt.title("Histogram Equalization")
<matplotlib.text.Text object at 0x7fc47f7f7d68>
>>> plt.imshow(mcH, vmin= 0, vmax = 255, interpolation = "none")
<matplotlib.image.AxesImage object at 0x7fc47f7eef28>
>>> plt.show()

