
# =========================================
# Image Noise Removal using OpenCV
# =========================================
# Aim: Remove Salt & Pepper and Gaussian noise from images
#      using Median, Gaussian, and Bilateral filters.
# =========================================

# =========================================
# STEP 1 – Import Libraries
# =========================================
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================
# STEP 2 – Load Original Image
# =========================================
img = cv2.imread("images/original/landscape.png")
img = cv2.resize(img, (512, 512))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Image loaded. Shape:", img.shape)

# =========================================
# STEP 3 – Add Salt & Pepper Noise
# =========================================
def add_salt_pepper(image, amount=0.05):
    noisy = image.copy()
    # Salt (white pixels)
    num_salt = int(image.size * amount * 0.5)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    # Pepper (black pixels)
    num_pepper = int(image.size * amount * 0.5)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

sp_noisy = add_salt_pepper(img, amount=0.05)
print("Salt & Pepper noise added (5%)")

# =========================================
# STEP 4 – Add Gaussian Noise
# =========================================
def add_gaussian_noise(image, sigma=25):
    noise = np.random.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)

gauss_noisy = add_gaussian_noise(img, sigma=25)
print("Gaussian noise added (sigma=25)")

# =========================================
# STEP 5 – Apply Filters on Salt & Pepper Noisy Image
# =========================================
sp_median   = cv2.medianBlur(sp_noisy, 5)
sp_gaussian = cv2.GaussianBlur(sp_noisy, (5, 5), 0)
sp_bilateral = cv2.bilateralFilter(sp_noisy, 9, 75, 75)
print("Filters applied on Salt & Pepper noisy image")

# =========================================
# STEP 6 – Apply Filters on Gaussian Noisy Image
# =========================================
g_median    = cv2.medianBlur(gauss_noisy, 5)
g_gaussian  = cv2.GaussianBlur(gauss_noisy, (5, 5), 0)
g_bilateral = cv2.bilateralFilter(gauss_noisy, 9, 75, 75)
print("Filters applied on Gaussian noisy image")

# =========================================
# STEP 7 – Calculate PSNR (Quality Metric)
# =========================================
print("\n--- PSNR Values (higher = better) ---")
print(f"Salt & Pepper Noisy:      {cv2.PSNR(img, sp_noisy):.2f} dB")
print(f"  → Median Filter:        {cv2.PSNR(img, sp_median):.2f} dB")
print(f"  → Gaussian Filter:      {cv2.PSNR(img, sp_gaussian):.2f} dB")
print(f"  → Bilateral Filter:     {cv2.PSNR(img, sp_bilateral):.2f} dB")
print()
print(f"Gaussian Noisy:           {cv2.PSNR(img, gauss_noisy):.2f} dB")
print(f"  → Median Filter:        {cv2.PSNR(img, g_median):.2f} dB")
print(f"  → Gaussian Filter:      {cv2.PSNR(img, g_gaussian):.2f} dB")
print(f"  → Bilateral Filter:     {cv2.PSNR(img, g_bilateral):.2f} dB")

# =========================================
# STEP 8 – Display Salt & Pepper Results
# =========================================
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

titles = ["Original", "S&P Noisy", "Median Filter", "Gaussian Filter", "Bilateral Filter"]
images = [img, sp_noisy, sp_median, sp_gaussian, sp_bilateral]

plt.figure(figsize=(20, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(to_rgb(images[i]))
    plt.title(titles[i])
    plt.axis("off")
plt.suptitle("Salt & Pepper Noise Removal", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("output_salt_pepper.png", dpi=150)
print("Salt & Pepper results saved to output_salt_pepper.png")

# =========================================
# STEP 9 – Display Gaussian Noise Results
# =========================================
titles2 = ["Original", "Gaussian Noisy", "Median Filter", "Gaussian Filter", "Bilateral Filter"]
images2 = [img, gauss_noisy, g_median, g_gaussian, g_bilateral]

plt.figure(figsize=(20, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(to_rgb(images2[i]))
    plt.title(titles2[i])
    plt.axis("off")
plt.suptitle("Gaussian Noise Removal", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("output_gaussian.png", dpi=150)
print("Gaussian results saved to output_gaussian.png")

# =========================================
# STEP 10 – Conclusion
# =========================================
print("\n=========================================")
print("Conclusion:")
print("=========================================")
print("- Median Filter works best for Salt & Pepper noise.")
print("- Bilateral Filter works best for Gaussian noise")
print("  (preserves edges while removing noise).")
print("- Gaussian Filter is fast but blurs edges.")
print("=========================================")
