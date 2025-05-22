import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_8_neighbors(y, x, height, width):
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbors.append((ny, nx))
    return neighbors

def region_growing(image, threshold):
    height, width = image.shape
    segmented_image = np.zeros((height, width), dtype=np.uint16)
    current_label = 1

    for r_idx in range(height):
        for c_idx in range(width):
            if segmented_image[r_idx, c_idx] == 0:
                seed_y, seed_x = r_idx, c_idx
                
                stack = [(seed_y, seed_x)]
                
                region_sum = float(image[seed_y, seed_x])
                region_count = 1
                
                segmented_image[seed_y, seed_x] = current_label
                
                while stack:
                    y, x = stack.pop() 

                    current_region_mean = region_sum / region_count
                    
                    for ny, nx in get_8_neighbors(y, x, height, width):
                        if segmented_image[ny, nx] == 0:
                            neighbor_val = float(image[ny, nx])
                            if abs(neighbor_val - current_region_mean) <= threshold:
                                segmented_image[ny, nx] = current_label
                                stack.append((ny, nx))
                                
                                region_sum += neighbor_val
                                region_count += 1
                
                if region_count > 0: 
                    current_label += 1
                    if current_label > 65530:
                        print(f"Warning: Max region labels {current_label} reached. Stopping early.")
                        return segmented_image 
    return segmented_image

def visualize_segmentation_labels(segmented_image_labels):
    height, width = segmented_image_labels.shape
    max_label = np.max(segmented_image_labels)
    
    output_vis = np.zeros((height, width, 3), dtype=np.uint8)
    
    if max_label == 0:
        return output_vis

    colors = np.random.randint(50, 255, size=(int(max_label) + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    for r in range(height):
        for c in range(width):
            label = segmented_image_labels[r, c]
            if label > 0 and label <= max_label:
                 output_vis[r, c, :] = colors[label]
            
    return output_vis


def main():
    input_dir = "data"
    output_dir_base = "output/Region-Growth"
    
    ensure_dir_exists(output_dir_base)

    image_paths = []
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_paths.append(os.path.join(input_dir, f))
    
    if not image_paths:
        print(f"No images found in {input_dir} with extensions .png, .jpg, .jpeg, .bmp, .tif, .tiff")
        if os.path.exists(input_dir):
            print(f"Contents of {input_dir}: {os.listdir(input_dir)}")
        else:
            print(f"Input directory {input_dir} does not exist.")
        return

    thresholds_to_test = [10, 20, 30, 50]

    for img_path in image_paths:
        print(f"Processing {img_path}...")
        try:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to load image: {img_path}. Skipping.")
                continue
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            continue

        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Region Growing Segmentation - {base_filename}", fontsize=16)
        axes = axes.ravel()

        for idx, threshold_val in enumerate(thresholds_to_test):
            print(f"  Applying Region Growth with threshold: {threshold_val}")
            
            segmented_labels = region_growing(image, threshold_val)
            
            output_visual = visualize_segmentation_labels(segmented_labels)
            
            output_visual_rgb = cv2.cvtColor(output_visual, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(output_visual_rgb)
            axes[idx].set_title(f"Threshold = {threshold_val}")
            axes[idx].axis('off')
        
        output_filename = f"{base_filename}_combined_thresholds.png"
        output_path = os.path.join(output_dir_base, output_filename)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"    Saved combined segmented image to {output_path}")

    print("\nRegion growing processing complete.")
    print(f"Output images are saved in: {output_dir_base}")

if __name__ == "__main__":
    main()

