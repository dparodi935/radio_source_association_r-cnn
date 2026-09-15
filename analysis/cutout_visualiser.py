""" Takes the .npy files in a selected folder and visualises them. Use any key to quickly flick through each cutout
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
cutout_folder = os.path.join(script_dir, "..", "cnn_data", "cutouts", "170_49_sigma3_rot_radio3_200")

# 1. Gather all the valid .npy files into a list first
files = [f for f in os.listdir(cutout_folder) if f.endswith('.npy')][:50]

if not files:
    print("No .npy files found.")
    sys.exit()

# 2. Set up the figure and a tracking index
fig, ax = plt.subplots()
current_idx = 0

# 3. Define a function to update the plot data         
def draw_image(idx):
    ax.clear() # Clear the previous image
    filepath = os.path.join(cutout_folder, files[idx])
    data = np.load(filepath)[0]
    
    ax.imshow(data, "inferno", origin="lower", vmin=0, vmax=1)
    # Added a counter to the title so yo       u know where you are
    ax.set_title(f"{files[idx].split("_r0.npy")[0]}")        
    
    if idx == 26:
        fig.savefig(f"{os.path.join(script_dir, "..", "cnn_data","image.png")}",dpi=180)
    
    fig.canvas.draw()
    

# 4. Define what happens when a key is pressed
def on_key_press(event):
    global current_idx
    
    if event.key == ' ':  # Spacebar
        current_idx += 1
        
        # Check if we reached the end
        if current_idx >= len(files):
            print("Finished showing all images.")
            plt.close()
        else:
            draw_image(current_idx)
            
    elif event.key == 'escape': # Optional: Press Esc to quit early
        plt.close()

# 5. Connect the key press event to our function 
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Draw the first image and start the Matplotlib window
draw_image(current_idx)
plt.show()