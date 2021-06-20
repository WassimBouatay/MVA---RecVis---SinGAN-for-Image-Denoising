import numpy as np
import matplotlib.pylab as plt
from skimage.transform import resize
import imageio
from os import walk
from skimage.restoration import estimate_sigma
import bm3d


noisy_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/SP_Noise/'
BM3D_path = 'C:/Files/M2 MVA/S1/Object recognition/Project/SinGAN-master/Input/BM3D/'


_, _, filenames = next(walk(noisy_path))

for image_name in filenames:
    print(image_name)
    noisy = imageio.imread(noisy_path+image_name)/255
    plt.imshow(noisy)
    plt.show()
    
    sigma_estimation = np.mean(estimate_sigma(noisy, multichannel=True))
    denoised = bm3d.bm3d(noisy, sigma_psd=sigma_estimation, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    plt.imshow(denoised)
    plt.show()  
    imageio.imwrite(BM3D_path+image_name, denoised)
    


