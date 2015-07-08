# Summary: Use some level of contrast adjustment, either mean or half the mean.
# Anything else will robably be unnecesary, plus most times raw data is feeded to
# the network and it still works. I do believe contrast will help, though because 
# otherwise the background mix with the image and features are harder to pick.
# Plus, it is common practie in radiology.

import scipy.misc
import matplotlib.pyplot as plt

#################### Background reduction (Contrast change)
#Background makes only the clusters light up, everything else goes to black.
# But does the background give info t the network (like breast density).

# vmin is done like this
mcB = mc.copy()
mcB[mcB <= 50] = 0
mcB[mcB>50] -= 50
# I'm not sure about vmax but is probably something like 
mcB[mcB > 200] = 0

## Actual code
>>> mc = scipy.misc.imread("breastMicrocalcification.png")
>>> mc = mc[:,:,0]
>>> np.mean(mc)
84.898947368421048
# For mcb, anything less than the average pixel value is considered background
>>> mcB = mc.copy()
>>> mcB[mcB <= 85] = 0
>>> mcB[mcB> 85] -= 85
# Same for half the mean pixel value
>>> mcB2 = mc.copy()
>>> mcB2[mcB2 <= 42] = 0
>>> mcB2[mcB2 > 42] -=42
# One where the background is set to black, but everything else is the same.
>>> mcB3 = mc.copy()
>>> mcB3[mcB3 <= 85] = 0
# Save em
>>> scipy.misc.imsave("mcB.png", mcB)
>>> scipy.misc.imsave("mcB2.png", mcB2)
>>> scipy.misc.imsave("mcB3.png", mcB3)
# Plot em
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7f139bd5a908>
>>> plt.imshow(mc)
<matplotlib.image.AxesImage object at 0x7f139c29fb38>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7f139c9b7e48>
>>> plt.imshow(mcB)
<matplotlib.image.AxesImage object at 0x7f139c9b7668>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7f139bdf1080>
>>> plt.imshow(mcB2)
<matplotlib.image.AxesImage object at 0x7f139be5ca20>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7f139bfc87f0>
>>> plt.imshow(mcB3)
<matplotlib.image.AxesImage object at 0x7f139bf90470>
>>> plt.show() 

# For masses
>>> bm = scipy.misc.imread("breastMass.png")
>>> bm = bm[:,:,0]
>>> np.mean(bm)
104.82896090534979


>>> bmB = bm.copy()
>>> bmB[bmB <= 105] = 0
>>> bmB[bmB> 105] -= 105

>>> bmB2 = bm.copy()
>>> bmB2[bmB2 <= 52] = 0
>>> bmB2[bmB2 > 52] -=52

>>> bmB3 = bm.copy()
>>> bmB3[bmB3 <= 105] = 0

>>> scipy.misc.imsave("bmB.png", bmB)
>>> scipy.misc.imsave("bmB2.png", bmB2)
>>> scipy.misc.imsave("bmB3.png", bmB3)

>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7f139c1bf8d0>
>>> plt.imshow(bm)
<matplotlib.image.AxesImage object at 0x7f139c9e24e0>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7f139be4f748>
>>> plt.imshow(bmB)
<matplotlib.image.AxesImage object at 0x7f139c2aa7b8>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7f139c9bcd30>
>>> plt.imshow(bmB2)
<matplotlib.image.AxesImage object at 0x7f139c2aae48>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7f139c9c2518>
>>> plt.imshow(bmB3)
<matplotlib.image.AxesImage object at 0x7f139be6d860>
>>> plt.show()

###################### Filters ##################
# Also played with other filters
# None worked.

import scipy.ndimage

# Gaussian
>>> gaussian_mc = scipy.ndimage.gaussian_filter(mc, sigma = 2)
# Median
>>> median_mc = scipy.ndimage.median_filter(mc,3)
# Sharpen
>>> filter_gaussian = scipy.ndimage.gaussian_filter(gaussian_mc, 1)
>>> sharpen_mc = gaussian_mc + 10*(gaussian_mc - filter_gaussian)
# DoG
>>> gaussian2_mc = scipy.ndimage.gaussian_filter(mc, sigma= 2)
>>> gaussian1_mc = scipy.ndimage.gaussian_filter(mc, sigma= 1)
>>> gaussiandiff_mc = gaussian2_mc-gaussian1_mc
# Plot
>>> plt.subplot(2,2,1)
<matplotlib.axes.AxesSubplot object at 0x7f139c9e7f60>
>>> plt.imshow(gaussian_mc)
<matplotlib.image.AxesImage object at 0x7f13a5a8cb70>
>>> plt.subplot(2,2,2)
<matplotlib.axes.AxesSubplot object at 0x7f139be7a048>
>>> plt.imshow(median_mc)
<matplotlib.image.AxesImage object at 0x7f139be7a3c8>
>>> plt.subplot(2,2,3)
<matplotlib.axes.AxesSubplot object at 0x7f139c0e74a8>
>>> plt.imshow(sharpen_mc)
<matplotlib.image.AxesImage object at 0x7f139c0e7320>
>>> plt.subplot(2,2,4)
<matplotlib.axes.AxesSubplot object at 0x7f139c0d0198>
>>> plt.imshow(gaussiandiff_mc)
<matplotlib.image.AxesImage object at 0x7f139bd3a470>
>>> plt.show()


# matshow vs imshow. Matshow to actually see the values with no interpolation or weird things (not quite so). 
#Don't know why the bm with a white arrow looks different than with a black arrow. Values are the same, other than the arrow.

mc = scipy.misc.imread("breastMicrocalcification.png")
mc = mc[:,:,0]
mcB = scipy.misc.imread("breastMicrocalcification.png")
mcB = mcB[:,:,0]
mcB[mcB < 50] = 0
plt.subplot(2,2,1)
plt.imshow(mc)
plt.subplot(2,2,2)
plt.imshow(mcB)
plt.subplot(2,2,3)
plt.imshow(mc, vmin = 40, vmax = 200)
plt.subplot(2,2,4)
plt.imshow(mcB, vmin = 40, vmax = 200)
plt.show()

# And for masses
bm = scipy.misc.imread("breastMass.png")
bm = bm[:,:,0]
bmB = scipy.misc.imread("breastMass.png")
bmB = bmB[:,:,0]
bmB[bmB < 50] = 0
plt.subplot(2,2,1)
plt.imshow(bm)
plt.subplot(2,2,2)
plt.imshow(bmB)
plt.subplot(2,2,3)
plt.imshow(bm, vmin = 40, vmax = 200)
plt.subplot(2,2,4)
plt.imshow(bmB, vmin = 40, vmax = 200)
plt.show()
#same for bm2 and bm3






