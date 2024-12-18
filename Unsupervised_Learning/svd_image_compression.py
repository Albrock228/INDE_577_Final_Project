import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from scipy.linalg import svd

def load_and_display_image(image_path):
    """Load and display the original grayscale image."""
    img = io.imread(image_path, as_gray=True)
    print(f"Original Image Shape: {img.shape}")

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    return img

def svd_compression(img, ranks):
    """Compress the image using SVD with specified ranks."""
    U, S, VT = svd(img, full_matrices=False)

    compressed_images = []
    for r in ranks:
        # Low-rank approximation
        img_compressed = np.dot(U[:, :r], np.dot(np.diag(S[:r]), VT[:r, :]))
        compressed_images.append((r, img_compressed))

    return compressed_images

def display_compressed_images(compressed_images):
    """Display compressed images with their ranks."""
    plt.figure(figsize=(12, 6))
    for i, (r, img) in enumerate(compressed_images, 1):
        plt.subplot(1, len(compressed_images), i)
        plt.imshow(img, cmap="gray")
        plt.title(f"Rank {r}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the example image
    image_path = "https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png"
    img = load_and_display_image(image_path)

    # Specify ranks for compression
    ranks = [10, 50, 100]

    # Perform SVD-based compression
    compressed_images = svd_compression(img, ranks)

    # Display the compressed images
    display_compressed_images(compressed_images)
