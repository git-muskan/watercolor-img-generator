import cv2
import numpy as np

image_path = r"C:\Users\sshar\OneDrive\Pictures\test.jpg"
paper_texture_path = r"C:\Users\sshar\OneDrive\Pictures\texture.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load main image.")
else:
    image_resized = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)  # Resize to desired dimensions
    image_cleared = cv2.medianBlur(image_resized, 7)  # Larger kernel size for more smoothing
    for _ in range(2):
        image_cleared = cv2.medianBlur(image_cleared, 7)
    
    # applying edge-preserving filter
    image_filtered = cv2.bilateralFilter(image_cleared, 9, 75, 75)
    for _ in range(2):
        image_filtered = cv2.bilateralFilter(image_filtered, 9, 100, 100)

    edges = cv2.Canny(image_filtered, 50, 150)  
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges = cv2.GaussianBlur(edges, (3, 3), 0) 

    image_with_edges = cv2.addWeighted(image_filtered, 0.8, edges, 0.2, 0)  # Reduced edge impact

    #applying Gaussian blur and enhancement
    gaussian_mask = cv2.GaussianBlur(image_with_edges, (5, 5), 1)  # Smaller kernel size and sigma
    image_enhanced = cv2.addWeighted(image_with_edges, 1.1, gaussian_mask, -0.1, 0)  # Less enhancement

    paper_texture = cv2.imread(paper_texture_path)

    if paper_texture is None:
        print("Error: Could not load paper texture image.")
    else:
        paper_texture = cv2.resize(paper_texture, (image_enhanced.shape[1], image_enhanced.shape[0]))

        gray_texture = cv2.cvtColor(paper_texture, cv2.COLOR_BGR2GRAY)
        _, thresh_texture = cv2.threshold(gray_texture, 240, 255, cv2.THRESH_BINARY_INV)  
        paper_texture = cv2.bitwise_and(paper_texture, paper_texture, mask=thresh_texture)

        image_with_texture = cv2.addWeighted(image_enhanced, 0.95, paper_texture, 0.05, 0) 

        max_width = 800
        max_height = 600

        height, width = image_with_texture.shape[:2]
        if width > max_width or height > max_height:
            scaling_factor = min(max_width / width, max_height / height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            image_with_texture = cv2.resize(image_with_texture, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # finally, displaying the images
        cv2.imshow('Original Image', image_resized)
        cv2.imshow('Watercolor Effect', image_with_texture)
        cv2.imwrite('watercolor_art.jpg', image_with_texture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
