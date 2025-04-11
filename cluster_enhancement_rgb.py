from cluster_enhancement import cluster_enhancement
import numpy as np
from PIL import Image

def enhance_image(image):
    image_array = np.array(image)
    for i in range(3):
        image_array[:, :, i] = cluster_enhancement(image_array[:, :, i])
        print("Enhancing channel", i + 1)
    image_array = Image.fromarray(image_array.astype(np.uint8))
    return image_array

def test():
    image = Image.open('image_rgb/test_rgb_2.jpg').convert('RGB')
    image = image.resize((256, 256))
    image.save('result/cluster_enhancement_rgb_2_original.png')
    output = enhance_image(image)
    output.save('result/cluster_enhancement_rgb_2_output.png')
    print("Enhanced image saved as 'result/cluster_enhancement_rgb_output.png'")