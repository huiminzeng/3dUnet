# 3dUnet

1. dataloader.py:
    - need to save preprocessed data at the directory
    
2. model_new.py:
    - 3D Unet model (reference https://github.com/MIC-DKFZ/ACDC2017)
   
3. loss.py:
    - cross entropy loss
    - dice loss

4. preprocess.py and preprocess.ipynb:
    - preprocessing raw data to generate numpy array files (input images and semantic segmentation masks)

5. train.py:
    - perform training and validation 

6. utils.py:
    - data augumentation
