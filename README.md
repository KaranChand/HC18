# Image segmentation for estimating fetal head circumference using U-net

U-net implementation original author and repo: https://github.com/zhixuhao/unet

--- 

Run model by running `main.py` after adjusting parameters and the output segmentations are created in `data/HC18/test_set/results`.
Post-processing and ellipse fitting is automatically also ran, the results of which can be found in `data/HC18/test_set/results_openclose` and `data/HC18/test_set/results_masked`, respectively.

Getting the csv with ellipse descriptors is done by running `ellipse_fit.ipynb'

--- 
Pre-trained U-net model we used for segmentation (500 steps/epoch, 10 epochs) is available here: https://we.tl/t-XQ02jrhBzG

Pre-trained nnU-net model we used is available here: https://we.tl/t-ypyihrv10R
