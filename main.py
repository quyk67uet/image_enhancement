import numpy as np
from PIL import Image 
from cluster_enhancement import cluster_enhancement

image = Image.open('image/aa.webp').convert('L')
image = image.resize((256, 256))
image.save('result/cluster_enhancement_original.png')
image = np.array(image)
output = cluster_enhancement(image)
output = Image.fromarray(output)
output.save('result/cluster_enhancement_output.png')