from cluster_enhancement import cluster_enhancement
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def enhance_image(image, kernel_size=27, alpha=0.5, k=5, p_value=0.15):
    image_array = np.array(image)
    for i in range(3):
        image_array[:, :, i] = cluster_enhancement(image_array[:, :, i], kernel_size=kernel_size, alpha=alpha, k=k, p_value=p_value)
        print("Enhancing channel", i + 1)
    image_array = Image.fromarray(image_array.astype(np.uint8))
    return image_array

def plot_channels(image):
    channels = ['Red', 'Green', 'Blue']
    for i, channel in enumerate(channels):
        plt.subplot(1, 4, i + 1)
        plt.imshow(image[:, :, i], cmap='gray')
        plt.title(f'{channel} Channel')
        plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.title(f'Final result')
    plt.axis('off')
    

def test():
    image = Image.open('image_rgb/test_rgb_2.jpg').convert('RGB')
    image = image.resize((256, 256))
    image.save('result/cluster_enhancement_rgb_2_original.png')
    output = enhance_image(image)
    output.save('result/cluster_enhancement_rgb_2_output.png')
    print("Enhanced image saved as 'result/cluster_enhancement_rgb_output.png'")

# image = Image.open('image_rgb/test_rgb_2.jpg').convert('RGB')
# image = image.resize((256, 256))
# output = enhance_image(image)
# plt.figure(1)
# plot_channels(np.array(image))
# plt.figure(2)
# plot_channels(np.array(output))
# plt.show()