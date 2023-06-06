# Image segmentation for estimating fetal head circumference using U-net

U-net implementation original author and repo: https://github.com/zhixuhao/unet

nnU-net implementation author and repo: https://github.com/MIC-DKFZ/nnUNet

--- 

Run U-net model by running `main.py` after adjusting parameters and the output segmentations are created in `data/HC18/test_set/results`.
Post-processing and ellipse fitting is automatically also ran, the results of which can be found in `data/HC18/test_set/results_openclose` and `data/HC18/test_set/results_masked`, respectively.

`nnUNet_HC18.ipynb` runs you through the setup process of nnU-net. The actual model was trained on Snellius, for which we give a download link below (+/- 260MB). Running inference is done by the command: `nnUNetv2_find_best_configuration 27 -c 2d `, as described on the Github page. I suggest to look there for further instructions.

Getting the csv with ellipse descriptors is done by running `ellipse_fit.ipynb'. Some adjustments have to be made for the nnU-net file system (in comments).

--- 
Pre-trained U-net model we used for segmentation (500 steps/epoch, 10 epochs) is available here: https://we.tl/t-XQ02jrhBzG

Pre-trained nnU-net model we used is available here: https://we.tl/t-ypyihrv10R
