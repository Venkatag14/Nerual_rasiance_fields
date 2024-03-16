import math
import kornia

def ssim(img1, img2):


    # Initialize SSIM loss module
    h = int(math.sqrt(img1.shape[0]))
    img1 = img1.view(1, 3, h, h)
    img2 = img2.view(1, 3, h, h)
    ssim_value = kornia.losses.ssim_loss(img1, img2, window_size=5)

    # Calculate SSIM
    # ssim_value = ssim_loss(img1, img2)

    return ssim_value.item()

def psnr(img1, img2):
    # Initialize PSNR loss module
    psnr_value = kornia.losses.psnr_loss(img1, img2)

    return psnr_value.item()