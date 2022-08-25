# SEAformer: Selective Edge Aggregation Transformer for 2D Medical Image Segmentation

## Code organiztion
- SEAformer_Keras.py: Keras version of SEAformer
- utils.py: Dependency functions required by SEAformer
- loss_function.py: dice_loss, jaccard_loss (IoU loss), BCE_IoU loss and IoU_Edge loss
- metrics.py: spec, sens, acc, iou, dice
- image: Qualitative analysis results of experiments

## Requirements
- CUDA/CUDNN
- TensorFlow-GPU and Keras > 2.7.0

## SEAformer overview
![Image text](https://github.com/Cjl-MedSeg/SEAformer/blob/SEAformer/Image/SEAformer_overview.jpg)
![Image text](https://github.com/Cjl-MedSeg/SEAformer/blob/SEAformer/Image/SEA_module.jpg)