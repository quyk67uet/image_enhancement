import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from cluster_enhancement import cluster_enhancement, mean_neigborhood, variance_neigborhood

def create_output_dirs():
    """Create necessary output directories for results"""
    dirs = [
        'results',
        'results/enhanced_images', 
        'results/comparison_figures',
        'results/zoom_comparisons',
        'results/mask_visualizations'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def unsharp_masking_gaussian(image, sigma=1.0, s=0.5, p_value=5):
    """
    Unsharp Masking with Gaussian Filter
    
    Parameters:
    image: Input grayscale image
    sigma: Standard deviation for Gaussian blur
    s: Scaling factor for subtraction
    p_value: Percentile value for intensity clipping
    
    Returns:
    Enhanced image using UM-Gaussian approach
    """
    # Ensure image is float32 for processing
    I = np.array(image, dtype=np.float32)
    
    # Create the mask using Gaussian blur
    kernel_size = int(2 * round(3 * sigma) + 1)  # Ensure odd size based on sigma
    Im_gauss = cv.GaussianBlur(I, (kernel_size, kernel_size), sigma)
    
    # Apply unsharp masking
    I_o = I - s * Im_gauss
    
    # Apply percentile-based clipping
    first_p = np.percentile(I_o, p_value)
    last_p = np.percentile(I_o, 100 - p_value)
    result = I_o.copy()
    result[result < first_p] = first_p
    result[result > last_p] = last_p
    
    # Normalize to 0-255 range
    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = result.astype(np.uint8)
    
    return result, Im_gauss

def unsharp_masking_median(image, window_size=15, s=0.5, p_value=5):
    """
    Unsharp Masking with Median Filter
    
    Parameters:
    image: Input grayscale image
    window_size: Size of the window for median filtering (must be odd)
    s: Scaling factor for subtraction
    p_value: Percentile value for intensity clipping
    
    Returns:
    Enhanced image using UM-Median approach
    """
    # Ensure image is float32 for processing
    I = np.array(image, dtype=np.float32)
    
    # Create the mask using Median filter
    Im_median = cv.medianBlur(I.astype(np.uint8), window_size).astype(np.float32)
    
    # Apply unsharp masking
    I_o = I - s * Im_median
    
    # Apply percentile-based clipping
    first_p = np.percentile(I_o, p_value)
    last_p = np.percentile(I_o, 100 - p_value)
    result = I_o.copy()
    result[result < first_p] = first_p
    result[result > last_p] = last_p
    
    # Normalize to 0-255 range
    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = result.astype(np.uint8)
    
    return result, Im_median

def anisotropic_diffusion(image, num_iter=10, K=20, g_func='exp', s=0.5, p_value=5):
    """
    Anisotropic Diffusion (AD) based Enhancement (Perona & Malik, 1990)
    
    Parameters:
    image: Input grayscale image
    num_iter: Number of iterations for diffusion
    K: Contrast parameter (edge threshold)
    g_func: Diffusion function ('exp' or 'frac')
    s: Scaling factor for subtraction
    p_value: Percentile value for intensity clipping
    
    Returns:
    Enhanced image using AD approach
    """
    # Ensure image is float32 for processing
    I = np.array(image, dtype=np.float32)
    
    # Make a copy for diffusion processing
    Im_ad = I.copy()
    
    # Define diffusion coefficient functions
    def g1(x, K):
        return np.exp(-(x/K)**2)
    
    def g2(x, K):
        return 1.0 / (1.0 + (x/K)**2)
    
    # Choose the appropriate function
    g = g1 if g_func == 'exp' else g2
    
    # Perform diffusion iterations
    dx = 1.0
    dy = 1.0
    dd = np.sqrt(2.0)
    
    # Set time step for numerical stability (dt ≤ 0.25 for 4-neighbor scheme)
    dt = 0.25
    
    # Padding for border handling
    padded = np.pad(Im_ad, 1, mode='edge')
    
    for _ in range(num_iter):
        # Get padded copy
        padded_copy = padded.copy()
        
        # Calculate gradients along 4-neighbors
        north = padded_copy[:-2, 1:-1] - padded_copy[1:-1, 1:-1]
        south = padded_copy[2:, 1:-1] - padded_copy[1:-1, 1:-1]
        east = padded_copy[1:-1, 2:] - padded_copy[1:-1, 1:-1]
        west = padded_copy[1:-1, :-2] - padded_copy[1:-1, 1:-1]
        
        # Calculate diffusion coefficients
        c_north = g(np.abs(north), K)
        c_south = g(np.abs(south), K)
        c_east = g(np.abs(east), K)
        c_west = g(np.abs(west), K)
        
        # Update the diffused image
        padded[1:-1, 1:-1] = padded_copy[1:-1, 1:-1] + dt * (
            c_north * north + c_south * south + 
            c_east * east + c_west * west
        )
    
    # Extract result from padding
    Im_ad = padded[1:-1, 1:-1]
    
    # Apply unsharp masking
    I_o = I - s * Im_ad
    
    # Apply percentile-based clipping
    first_p = np.percentile(I_o, p_value)
    last_p = np.percentile(I_o, 100 - p_value)
    result = I_o.copy()
    result[result < first_p] = first_p
    result[result > last_p] = last_p
    
    # Normalize to 0-255 range
    result = (result - result.min()) / (result.max() - result.min()) * 255
    result = result.astype(np.uint8)
    
    return result, Im_ad

def process_image(img_path):
    """Process a single image with all enhancement methods"""
    # Read and convert image to grayscale
    print(f"\nProcessing image: {os.path.basename(img_path)}")
    image = Image.open(img_path).convert('L')
    image_np = np.array(image)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Define experiment parameters (adjust based on image characteristics)
    kernel_size = 15
    alpha = 0.5
    k_iterations = 5
    s_factor = 0.5
    p_value = 5
    
    # Method 1: UM-Gaussian
    print("  Applying UM-Gaussian...")
    gaussian_sigma = 3.0  # Adjust based on kernel_size equivalent smoothing
    um_gauss_result, gauss_mask = unsharp_masking_gaussian(
        image_np, sigma=gaussian_sigma, s=s_factor, p_value=p_value
    )
    
    # Method 2: UM-Median
    print("  Applying UM-Median...")
    median_window = kernel_size  # Use same kernel size as CF for fairness
    um_median_result, median_mask = unsharp_masking_median(
        image_np, window_size=median_window, s=s_factor, p_value=p_value
    )
    
    # Method 3: Anisotropic Diffusion
    print("  Applying Anisotropic Diffusion...")
    ad_result, ad_mask = anisotropic_diffusion(
        image_np, num_iter=15, K=20, g_func='exp', s=s_factor, p_value=p_value
    )
    
    # Method 4: Clustering Filter (from imported module)
    print("  Applying Clustering Filter...")
    cf_result = cluster_enhancement(
        image_np, kernel_size=kernel_size, alpha=alpha, k=k_iterations, p_value=p_value
    )
    
    # Log parameters used
    print(f"  Parameters used:")
    print(f"    - Common: s={s_factor}, p_value={p_value}")
    print(f"    - UM-Gaussian: sigma={gaussian_sigma}")
    print(f"    - UM-Median: window_size={median_window}")
    print(f"    - Anisotropic Diffusion: iterations=15, K=20, g_func=exp")
    print(f"    - Clustering Filter: kernel_size={kernel_size}, alpha={alpha}, k={k_iterations}")
    
    # Save individual enhanced images
    cv.imwrite(f"results/enhanced_images/{img_name}_original.png", image_np)
    cv.imwrite(f"results/enhanced_images/{img_name}_um_gauss.png", um_gauss_result)
    cv.imwrite(f"results/enhanced_images/{img_name}_um_median.png", um_median_result)
    cv.imwrite(f"results/enhanced_images/{img_name}_ad.png", ad_result)
    cv.imwrite(f"results/enhanced_images/{img_name}_cf.png", cf_result)
    
    # Generate comparison figure
    create_comparison_figure(
        img_name, image_np, um_gauss_result, um_median_result, ad_result, cf_result
    )
    
    # Generate mask visualization
    create_mask_visualization(
        img_name, image_np, gauss_mask, median_mask, ad_mask
    )
    
    # Generate zoom comparisons
    create_zoom_comparisons(
        img_name, image_np, um_gauss_result, um_median_result, ad_result, cf_result
    )
    
    return {
        'original': image_np,
        'um_gauss': um_gauss_result,
        'um_median': um_median_result,
        'ad': ad_result,
        'cf': cf_result,
        'masks': {
            'gauss': gauss_mask,
            'median': median_mask,
            'ad': ad_mask
        }
    }

def create_comparison_figure(img_name, original, um_gauss, um_median, ad, cf):
    """Create and save side-by-side comparison of all methods"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(um_gauss, cmap='gray')
    axes[1].set_title('UM-Gaussian')
    axes[1].axis('off')
    
    axes[2].imshow(um_median, cmap='gray')
    axes[2].set_title('UM-Median')
    axes[2].axis('off')
    
    axes[3].imshow(ad, cmap='gray')
    axes[3].set_title('Anisotropic Diffusion')
    axes[3].axis('off')
    
    axes[4].imshow(cf, cmap='gray')
    axes[4].set_title('Clustering Filter')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/comparison_figures/{img_name}_comparison.png", dpi=300)
    plt.close()

def create_mask_visualization(img_name, original, gauss_mask, median_mask, ad_mask):
    """Create and save visualization of the generated masks"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(gauss_mask, cmap='gray')
    axes[1].set_title('Gaussian Mask')
    axes[1].axis('off')
    
    axes[2].imshow(median_mask, cmap='gray')
    axes[2].set_title('Median Mask')
    axes[2].axis('off')
    
    axes[3].imshow(ad_mask, cmap='gray')
    axes[3].set_title('AD Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/mask_visualizations/{img_name}_masks.png", dpi=300)
    plt.close()

def create_zoom_comparisons(img_name, original, um_gauss, um_median, ad, cf):
    """Create zoomed-in comparisons of critical regions"""
    # Identify interesting region (center 1/4 of the image as an example)
    h, w = original.shape
    center_h, center_w = h // 2, w // 2
    size_h, size_w = h // 4, w // 4
    
    zoom_region = (
        slice(center_h - size_h // 2, center_h + size_h // 2),
        slice(center_w - size_w // 2, center_w + size_w // 2)
    )
    
    # Create zoomed-in comparison
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(original[zoom_region], cmap='gray')
    axes[0].set_title('Original (Zoom)')
    axes[0].axis('off')
    
    axes[1].imshow(um_gauss[zoom_region], cmap='gray')
    axes[1].set_title('UM-Gaussian (Zoom)')
    axes[1].axis('off')
    
    axes[2].imshow(um_median[zoom_region], cmap='gray')
    axes[2].set_title('UM-Median (Zoom)')
    axes[2].axis('off')
    
    axes[3].imshow(ad[zoom_region], cmap='gray')
    axes[3].set_title('AD (Zoom)')
    axes[3].axis('off')
    
    axes[4].imshow(cf[zoom_region], cmap='gray')
    axes[4].set_title('CF (Zoom)')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/zoom_comparisons/{img_name}_zoom_center.png", dpi=300)
    plt.close()
    
    # Optional: Create a second zoom focusing on edges
    # Detect edges using Sobel filter
    edges = cv.Sobel(original, cv.CV_64F, 1, 1, ksize=3)
    edges = np.abs(edges)
    
    # Find region with strongest edges
    edge_sum = np.zeros_like(edges)
    window_size = 20
    for i in range(0, h - window_size, window_size):
        for j in range(0, w - window_size, window_size):
            edge_sum[i:i+window_size, j:j+window_size] = np.sum(
                edges[i:i+window_size, j:j+window_size]
            )
    
    edge_idx = np.unravel_index(np.argmax(edge_sum), edge_sum.shape)
    edge_region = (
        slice(max(0, edge_idx[0] - size_h // 2), min(h, edge_idx[0] + size_h // 2)),
        slice(max(0, edge_idx[1] - size_w // 2), min(w, edge_idx[1] + size_w // 2))
    )
    
    # Create edge-focused zoom comparison
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(original[edge_region], cmap='gray')
    axes[0].set_title('Original (Edge Zoom)')
    axes[0].axis('off')
    
    axes[1].imshow(um_gauss[edge_region], cmap='gray')
    axes[1].set_title('UM-Gaussian (Edge Zoom)')
    axes[1].axis('off')
    
    axes[2].imshow(um_median[edge_region], cmap='gray')
    axes[2].set_title('UM-Median (Edge Zoom)')
    axes[2].axis('off')
    
    axes[3].imshow(ad[edge_region], cmap='gray')
    axes[3].set_title('AD (Edge Zoom)')
    axes[3].axis('off')
    
    axes[4].imshow(cf[edge_region], cmap='gray')
    axes[4].set_title('CF (Edge Zoom)')
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/zoom_comparisons/{img_name}_zoom_edge.png", dpi=300)
    plt.close()

def process_all_images():
    """Process all images in the images directory"""
    # Create output directories
    create_output_dirs()
    
    # Get all image files
    image_dir = 'images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No image files found in directory: {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = {}
    for img_path in image_files:
        try:
            results[img_path] = process_image(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("\nAll image processing complete!")
    print(f"Results saved to the following directories:")
    print(f"  - Enhanced images: results/enhanced_images/")
    print(f"  - Comparison figures: results/comparison_figures/")
    print(f"  - Zoom comparisons: results/zoom_comparisons/")
    print(f"  - Mask visualizations: results/mask_visualizations/")

if __name__ == "__main__":
    start_time = time.time()
    process_all_images()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds") 