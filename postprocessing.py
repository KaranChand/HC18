from PIL import Image
import os 
import numpy as np

# processes the output from nnUnet

folder_path = "pred_nnUNet/"
out_folder_path = "postprocessed/"

# Get a list of all the files in the folder
file_list = os.listdir(folder_path)

for file_name  in file_list:
    file_path = os.path.join(folder_path, file_name)
    
    img = Image.open(file_path)
    np_img = np.array(img)
    np_img = np_img.astype(np.float32)

    # convert to 255
    np_img = np.where(np_img > 0, 255., 0)
    np_img = np_img.astype(np.uint8)

    img = Image.fromarray(np_img)
    
    out_file_path = os.path.join(out_folder_path, file_name)
    # save image to output path
    img.save(out_file_path, 'PNG')