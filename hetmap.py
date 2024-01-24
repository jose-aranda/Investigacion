import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from PIL import Image


#cover_img = plt.imread('jarra_jengibre_10.jpg')
#surface_df = pd.read_csv("output_20.csv")

csv_folder = "imagen_1"  # Replace with the path to your CSV files folder
#jpg_folder = "jpg"  # Replace with the path to your JPG files folder


# List files in both CSV and JPG folders
csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv")])
jpg_file = "cestreria_01.jpg"

# Ensure there are the same number of files in each folder
#if len(csv_files) != len(jpg_files):
#    print("Error: The number of CSV and JPG files doesn't match.")
#    exit()

def create_heatmap(surface_df,csv_file, cover_img, i):
    
    cover_img = plt.imread(cover_img)
    gaze_on_surf = pd.read_csv(surface_df)

    #gaze_on_surf = surface_df
    #gaze_on_surf = surface_df[surface_df.on_surf == True]
    #gaze_on_surf = surface_df[(surface_df.confidence > 0.8)]

    grid = cover_img.shape[0:2] # height, width of the loaded image
    heatmap_detail = 0.05 # this will determine the gaussian blur kerner of the image (higher number = more blur)

    #print(grid)

    gaze_on_surf_x = gaze_on_surf['x_norm']
    gaze_on_surf_y = gaze_on_surf['y_norm']
 
    # flip the fixation points
    # from the original coordinate system,
    # where the origin is at botton left,
    # to the image coordinate system,
    # where the origin is at top left
    gaze_on_surf_y = 1 - gaze_on_surf_y

    # make the histogram
    hist, x_edges, y_edges = np.histogram2d(
        gaze_on_surf_y,
        gaze_on_surf_x,
        range=[[0, 1.0], [0, 1.0]],
        #normed=False, 
        bins=grid
    )

    # gaussian blur kernel as a function of grid/surface size
    filter_h = int(heatmap_detail * grid[0]) // 2 * 2 + 1
    filter_w = int(heatmap_detail * grid[1]) // 2 * 2 + 1
    heatmap = gaussian_filter(hist, sigma=(filter_w, filter_h), order=0)
    
    #print(heatmap)
    print(type(heatmap))
    # Specify the file path and name
    file_path = 'heatmap.csv'
    # Save the array to a CSV file
    np.savetxt(file_path, heatmap, delimiter=',')
    
    
    # Step 1: Get height and width
    height, width = heatmap.shape

    # Step 2: Calculate remainder
    height_remainder = height % 16
    width_remainder = width % 16

    # Step 3: Eliminate last rows and columns
    new_height = height - height_remainder
    new_width = width - width_remainder

    # Update the array by keeping only the relevant portion
    your_array = heatmap[:new_height, :new_width]

    # Save the array to a CSV file
    np.savetxt(f"reminder_heatmap_{csv_file[:-4]}_img{i}.csv", your_array, delimiter=',')
    plt.figure(figsize=(8,8))
    plt.imshow(cover_img)
    plt.imshow(your_array, cmap='jet', alpha=0.5)
    plt.axis('off')
    #plt.savefig(f"{csv_file[:-4]}reminder_heatmap{i}.png");
    
    # display the histogram and reference image
    #print(i)
    np.savetxt(f"heatmap_{csv_file[:-4]}_img{i}.csv", your_array, delimiter=',')
    plt.figure(figsize=(8,8))
    plt.imshow(cover_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    #plt.savefig(f"{csv_file[:-4]}_heatmap_{i}.png");

    plt.pyplot.close()
# Process each pair of CSV and JPG files

for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)
        
    create_heatmap(csv_path,csv_file, jpg_file, jpg_file[:-4])
    
    print(csv_file)
    print(jpg_file)
    