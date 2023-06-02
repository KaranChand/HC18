from model import *
from data import *

# Data augmentation options
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# Train model
HC18 = trainGenerator(2,'data/HC18/training_set','image','label_filled',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_HC18.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(HC18,steps_per_epoch=50,epochs=3,callbacks=[model_checkpoint])

# Test model
HC18_test = testGenerator("data/HC18/test_set")
model = unet()
model.load_weights("unet_HC18-500-10.hdf5")
results = model.predict_generator(HC18_test,335,verbose=1)
saveResult("data/HC18/test_set/results",results)

# Resize to 800 x 540
resize_output_images('data/HC18/test_set/results/', (800, 540))

# Threshold and post-process with opening/closing
postprocess('data/HC18/test_set/results/', 200)
