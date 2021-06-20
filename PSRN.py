import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy import ndimage
import imageio

def psnr(clean, img):
    img = img
    clean = clean
    mse = np.mean((clean-img)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX/np.sqrt(mse))


input_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Set14/'
noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/GaussianNoise/'
output_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Output/Paint2image/'
filtered_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Filtered/'
NLmeans_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/NLmeans/Gaussian/'
BM3D_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/BM3D/'
p = "C:/Files/M2 MVA/"




#### Denoised image
denoised = imageio.imread(p+"ffdnet (9).png")/255
plt.show()

#### Clean image
image = imageio.imread(input_path+"zebra.png")/255
image = resize(image, denoised.shape, mode='reflect')
plt.imshow(image)
plt.show()

print("Denoised image PSNR", psnr(image, denoised))


"""
#### Noisy image
noisy = imageio.imread(noisy_path+'sigma=30-flowers.png')/255
noisy = resize(noisy, image.shape, mode='reflect')
plt.imshow(noisy)
plt.show()
print("Noisy image PSNR", psnr(image, noisy))


### Median filter
filtered = imageio.imread(filtered_path+"f-sigma=30-flowers.png")/255
filtered = resize(filtered, image.shape, mode='reflect')
print("Median-Filter denoised image PSNR", psnr(image, filtered))
plt.imshow(filtered)
plt.show()

#### SinGAN result
output = imageio.imread(output_path+'NL-sigma=30-flowers/start_scale=7.png')/255
#output = imageio.imread(output_path+'f-sigma=30-Lenna/start_scale=7.png')/255
output = resize(output, image.shape, mode='reflect')
plt.imshow(output)
plt.show()
print("sinGAN Denoised image PSNR", psnr(image, output))


#### NLmeans
NLmeans_denoised = imageio.imread(NLmeans_path+"NL-sigma=30-flowers.png")/255
NLmeans_denoised = resize(NLmeans_denoised, image.shape, mode='reflect')
print("NLmeans denoised image PSNR", psnr(image, NLmeans_denoised))
plt.imshow(NLmeans_denoised)
plt.show()

### BM3D
bm3d = imageio.imread(BM3D_path+"sigma=30-baboon.png")/255
bm3d = resize(bm3d, image.shape, mode='reflect')
print("BM3D denoised image PSNR", psnr(image, bm3d))
plt.imshow(bm3d)
plt.show()


input_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Set14/'
noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/SP_Noise/'
output_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Output/Paint2image/'
filtered_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/Filtered/'
NLmeans_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/NLmeans/Gaussian/'
BM3D_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/BM3D/'


#### Clean image
image = imageio.imread(input_path+"bridge.png")/255
plt.imshow(image)
plt.show()

#### Noisy image
noisy = imageio.imread(noisy_path+'SP-bridge.png')/255
noisy = resize(noisy, image.shape, mode='reflect')
plt.imshow(noisy)
plt.show()
print("Noisy image PSNR", psnr(image, noisy))


### Median filter
filtered = imageio.imread(filtered_path+"f-SP-bridge.png")/255
filtered = resize(filtered, image.shape, mode='reflect')
print("Median-Filter denoised image PSNR", psnr(image, filtered))
plt.imshow(filtered)
plt.show()


#### SinGAN result
output = imageio.imread(output_path+'f-SP-bridge/start_scale=7.png')/255
#output = imageio.imread(output_path+'f-sigma=30-Lenna/start_scale=7.png')/255
output = resize(output, image.shape, mode='reflect')
plt.imshow(output)
plt.show()
print("sinGAN Denoised image PSNR", psnr(image, output))



#### NLmeans
NLmeans_denoised = imageio.imread(NLmeans_path+"NL-sigma=30-bridge.png")/255
NLmeans_denoised = resize(NLmeans_denoised, image.shape, mode='reflect')
print("NLmeans denoised image PSNR", psnr(image, NLmeans_denoised))
plt.imshow(NLmeans_denoised)
plt.show()


### BM3D
bm3d = imageio.imread(BM3D_path+"SP-bridge.png")/255
bm3d = resize(bm3d, image.shape, mode='reflect')
print("BM3D denoised image PSNR", psnr(image, bm3d))
plt.imshow(bm3d)
plt.show()
"""