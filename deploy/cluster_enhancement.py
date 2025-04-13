import numpy as np
from PIL import Image 
import cv2 as cv
import matplotlib.pyplot as plt
import time
    
def exponetial_matrix(image, i, j, beta, kernel_size):
    padding = kernel_size // 2
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1].copy()
    cell_value = image[i, j].copy()
    dif = perception_view - cell_value
    return np.exp(-beta * dif ** 2)

def enhance_a_pixel(image, i, j, w, beta, kernel_size):
    padding = kernel_size // 2
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1].copy()
    exponetial = exponetial_matrix(image, i, j, beta, kernel_size)
    return np.sum(w * exponetial * perception_view) / np.sum(w * exponetial)

def calculate_w(alpha, kernel_size):
    padding = kernel_size // 2
    distance = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    for x in range(kernel_size):
        for y in range(kernel_size):
            distance[x, y] = np.sqrt((x - padding)**2 + (y - padding)**2)
    return np.exp(-alpha * distance)

def calculate_beta(image, i, j, w, kernel_size):
    padding = kernel_size // 2
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1].copy()
    
    y_macron = np.sum(w * perception_view) / np.sum(w)
    sigma_2 = np.sum(w * (perception_view - y_macron) ** 2) / np.sum(w)
    beta = 1 / (2 * sigma_2  + 1e-10)
    return beta

def cluster_filter(image, alpha, kernel_size):
    w = calculate_w(alpha, kernel_size)
    result = np.zeros(image.shape, dtype=np.float32)
    image = image.copy()
    image = np.pad(image, (kernel_size // 2, kernel_size // 2), mode='edge')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            beta = calculate_beta(image, i + kernel_size // 2, j + kernel_size // 2, w, kernel_size)
            result[i, j] = enhance_a_pixel(image, i + kernel_size // 2, j + kernel_size // 2, w, beta, kernel_size)
    return result

def mean_neigborhood(image, i, j, kernel_size):
    image = np.pad(image, (kernel_size // 2, kernel_size // 2), mode='edge')
    padding = kernel_size // 2
    i += padding
    j += padding
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1]
    return np.mean(perception_view)

def variance_neigborhood(image, i, j, kernel_size):
    image = np.pad(image, (kernel_size // 2, kernel_size // 2), mode='edge')
    padding = kernel_size // 2
    i += padding
    j += padding
    perception_view = image[i-padding:i+padding+1, j-padding:j+padding+1]
    return np.var(perception_view)

def cluster_enhancement(image, kernel_size=3, alpha=0.5, k=5, p_value=0.5):
    I = np.array(image, dtype=np.float32)
    # Step 1
    I_i = I.copy()
    for i in range(k):
        I_i = cluster_filter(I_i, alpha, kernel_size)

    # Step 2
    I_d = (I - I_i).copy()
    I_m = I.copy()
    # Step 3 + 4
    for i in range(I_d.shape[0]):
        for j in range(I_d.shape[1]):
            mean = mean_neigborhood(I_d, i, j, 40)
            variance = variance_neigborhood(I_d, i, j, 40)
            if abs(I_d[i, j] - mean) < 2.5 * variance:
                I_m[i, j] = I_i[i, j].copy()
    # Step 5
    I_o = (I - 0.5 * I_m).copy()

    # Step 6
    result = I_o.copy()

    first_ten = np.percentile(I_o, p_value)
    last_ten = np.percentile(I_o, 100 - p_value)
    result[result < first_ten] = first_ten
    result[result > last_ten] = last_ten    

    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = result.astype(np.uint8)
    return result

def test():
    image = Image.open('images/cameraman.png').convert('L')
    #image = image.resize((256, 256))
    image = np.array(image)
    output = cluster_enhancement(image, kernel_size=15, alpha=0.5, k=5, p_value=5)

    plt.figure(1)
    plt.imshow(image, cmap='gray')
    plt.title('original')

    plt.figure(2)
    plt.imshow(output, cmap='gray')
    plt.title('result')


    plt.figure(3)
    plt.hist(output.reshape(-1), bins=256)
    plt.title('hist_result')

    plt.figure(4)
    plt.hist(image.reshape(-1), bins=256)
    plt.title('hist_original')

    plt.show()

# Comment out test() call to avoid automatically running the test when importing the module
# test()