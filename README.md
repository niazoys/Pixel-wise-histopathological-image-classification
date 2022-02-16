# Pixel-wise-histopathological-image-classification

## Required Packages

- Numpy 
- Pytorch
- Torchvision
- Torchmetrics
- Sci-kit-Image
- Imageio
- Matplotlib 
- Tqdm
- Segmentation_models_pytorch

## Training the Network

- Use the **train.py** script.This script extracts the slices from the CT volume and train the model on those slices.
- Stores the result (training/validation loss accuracy,dice score and trained model weights) on the given output path.

- Arguments 
  - Output : output folder path for the results
  - Epoch : Number of epoch
  - Batch : Batch size
  - Learning : Learning rate for the networks
  - Network : select the network
  - Input size : image size (height * width) of input image
  - load : path of model weight to load
  

Command to run:
```
python train.py -o 'you_output_path' -e 100 -b 10 -l 0.0001 -n 'vgg19' -is 256 

```
 or
- Just run the file from IDE by setting the default values of the input argument.

 ## Prediction with the Network 
 
 - Use the **inference.py** python script to run inference.
 - Stores the result (Mask overlayed on image,precision,recall etc.) on a new folder to the path where the model weight loaded from.
 
 - Arguments 
    
    - Batch : Batch size 
    - Network : select the network
    - Input size : image size (height * width) of input image
    - load : path of model weight to load

Command to run:
```
python inference.py -b 10  -n  'vgg19'  -f  './../your_path_of_model_weight/weight.pth'

```
- Just run the file from IDE by setting the default values of the input argument.
