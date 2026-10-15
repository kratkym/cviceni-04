import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread 
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist,equalize_adapthist

plt.close('all')

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist, equalize_adapthist

plt.close('all')

# Ukol 1
# =============================================================================
# Nactete obraz "Matterhorn.jpg“, prevedte jej na sedotonovy, vytvorte histogram obrazu
# K obrazu prictete hodnotu 0.3 a zobrazte jej i s odpovidajicim histogramem
# Vytvorte dalsi obraz vynasobenim puvodniho obrazu hodnotou 1.5 a zobrazte jej i s odpovidajicim histogramem
# =============================================================================

# Nactete obraz "Matterhorn.jpg“, prevedte jej na sedotonovy, vytvorte histogram obrazu
img = rgb2gray(imread('data/Matterhorn.jpg'))

plt.figure()
plt.subplot(2,3,1)
plt.imshow(img,cmap='gray', vmin = 0, vmax = np.max(img))
plt.title('Originalni obraz')

plt.subplot(2,3,4)
plt.hist(img.ravel(), bins=256, range=(0, 1))
plt.title('Histogram originalniho obrazu')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,3,2)
img_plus = img + 0.3
plt.imshow(img_plus,cmap='gray', vmin = 0, vmax = np.max(img_plus))
plt.title('Obr + 0.3')

plt.subplot(2,3,5)
plt.hist(img_plus.ravel(), bins=256, range=(0, np.max(img_plus)))
plt.title('Histogram obrazu po pricitani')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,3,3)
img_mult = img * 1.5
plt.imshow(img_mult,cmap='gray', vmin = 0, vmax = np.max(img_mult))
plt.title('Obraz * 1.5')

plt.subplot(2,3,6)
plt.hist(img_mult.ravel(), bins=256, range=(0, np.max(img_mult)))
plt.title('Histogram obrazu po nasobeni')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.show()


# Ukol 2
# =============================================================================
# Nactete obraz „rtg_ruka.jpg“, prevedte jej do rozsahu 0-1 a upravte jej tak, aby byla vyuzita cela skala stupnu sedi
# Porovnejte obrazy a jejich histogramy
# Vytvorte negativ obrazu a zobrazte jej spolu s jeho histogramem
# Upravte kontrast obrazu pomoci gama transformace s hodnotou γ>1 a γ<1
# Provedte vhodnou korekci nelinearniho zkresleni v obraze „gamma.png“ pomoci gama korekce
# =============================================================================
plt.close('all')

# Nactete obraz „rtg_ruka.jpg“, prevedte jej do rozsahu 0-1 a upravte jej tak, aby byla vyuzita cela skala stupnu sedi
img = imread('data/rtg_ruka.jpg')

# Prevod do rozsahu 0-1
img = img.astype(float) / 255.0

# Roztazeni histogramu na celou skalu 0-1
img_min = np.min(img)
img_max = np.max(img)
img_transform = (img - img_min) / (img_max - img_min)

# Porovnejte obrazy a jejich histogramy
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img,cmap='gray', vmin=0,vmax=1)
plt.title('Originalni obraz')

plt.subplot(2,2,3)
plt.hist(img.ravel(), bins=256, range=(0, 1))
plt.title('Histogram originalniho obrazu')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,2,2)
plt.imshow(img_transform,cmap='gray',vmin=0,vmax=1)
plt.title('Transformovany obraz')

plt.subplot(2,2,4)
plt.hist(img_transform.ravel(), bins=256, range=(0, 1))
plt.title('Histogram transformovaneho obrazu')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')
plt.show()


# Negativ a gama transformace
img_neg = 1 - img_transform
img_gamma_less = img_transform ** 0.5
img_gamma_greater = img_transform ** 2

plt.figure()
plt.subplot(2,3,1)
plt.imshow(img_neg,cmap='gray',vmin=0,vmax=1)
plt.title('Negativ')

plt.subplot(2,3,4)
plt.hist(img_neg.ravel(), bins=256, range=(0, 1))
plt.title('Histogram negativu')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,3,2)
plt.imshow(img_gamma_less,cmap='gray',vmin=0,vmax=1)
plt.title('Transformovany obraz gamma < 1')

plt.subplot(2,3,5)
plt.hist(img_gamma_less.ravel(), bins=256, range=(0, 1))
plt.title('Histogram transformovaneho obrazu gamma < 1')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,3,3)
plt.imshow(img_gamma_greater,cmap='gray',vmin=0,vmax=1)
plt.title('Transformovany obraz gamma > 1')

plt.subplot(2,3,6)
plt.hist(img_gamma_greater.ravel(), bins=256, range=(0, 1))
plt.title('Histogram transformovaneho obrazu gamma > 1')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')
plt.show()


# Gamma korekce obrazu gamma.png
img_gamma = imread('data/gamma.png')

# Prevod do sedotonoveho obrazu a rozsahu 0-1
if img_gamma.ndim == 3:
    img_gamma = rgb2gray(img_gamma)

img_gamma = img_gamma.astype(float)

if np.max(img_gamma) > 1:
    img_gamma = img_gamma / 255.0

# Korekce tmaveho obrazu pomoci gamma < 1
img_gamma_corrected = img_gamma ** 0.5

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_gamma,vmin=0,vmax=1)
plt.title('Originalni obraz')

plt.subplot(1,2,2)
plt.imshow(img_gamma_corrected,vmin=0,vmax=1)
plt.title('Obr po gamma korekci')
plt.show()



# Ukol 3
# =============================================================================
# Nactete obraz "fotbal.png“, prevedte jej na sedotonovy o rozsahu 0-1
# Provedte ekvalizaci histogramu a zobrazte vysledny obraz s jeho histogramem (equalize_hist)
# Provedte kontrastne omezenou adaptivni ekvalizaci histogramu (CLAHE) a zobrazte vysledny obraz s jeho histogramem
# =============================================================================
plt.close('all')

# Nactete obraz "fotbal.png“, prevedte jej na sedotonovy o rozsahu 0-1
img = rgb2gray(imread('data/fotbal.png'))

# Ekvalizace histogramu a CLAHE
img_equalized = equalize_hist(img)
img_clahe = equalize_adapthist(img)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(img, vmin = 0, vmax = 1, cmap = 'gray')
plt.title('Originalni obraz')

plt.subplot(2,3,4)
plt.hist(img.ravel(), bins=256, range=(0, 1))
plt.title('Histogram originalu')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,3,2)
plt.imshow(img_equalized, vmin = 0, vmax = 1, cmap = 'gray')
plt.title('Ekvalizovany obraz')

plt.subplot(2,3,5)
plt.hist(img_equalized.ravel(), bins=256, range=(0, 1))
plt.title('Histogram ekvalizovaneho')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')

plt.subplot(2,3,3)
plt.imshow(img_clahe, vmin = 0, vmax = 1, cmap = 'gray')
plt.title('CLAHE')

plt.subplot(2,3,6)
plt.hist(img_clahe.ravel(), bins=256, range=(0, 1))
plt.title('Histogram CLAHE')
plt.xlabel('Jasova hodnota')
plt.ylabel('Cetnost')
plt.show()