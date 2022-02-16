
import torch
import os
import imageio
import numpy as np
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from skimage import  color
from torchmetrics.functional import recall,precision


def stats(preds,gt):
    ''' This method calculates the stats'''
    rec=recall(preds, gt, average='macro', num_classes=4,mdmc_average='global')
    prec = precision(preds, gt, average='macro', num_classes=4,mdmc_average='global')
    dice=dice_score(preds, torch.tensor(gt,dtype=torch.int64))
    return prec,rec,dice

def write_overlayed_image(original_image,preds,batch,out,groundtruth=None): 
    '''This method overlays the prediction mask on top of image'''
    
    # Selecting color explicitly here ensures that whatever the number of unique classes is the mask
    # each time a specific class gets the specific color
    for i in range(original_image.shape[0]):
        labels =np.unique(preds[i,:,:])
        apply_col=[]
        colors=[(100, 50, 200),(255, 0, 0),(0,255, 0),(0, 0, 255)]
        
        for j in labels:
            apply_col.append(colors[j])
        
        if groundtruth is not None:    
            img_pred=color.label2rgb(preds[i,:,:],original_image[i,:,:,:],colors=apply_col, alpha=0.0007)
            file_name_pred=str(batch)+'.png'
            img_mask=color.label2rgb(groundtruth[i,:,:],original_image[i,:,:,:],colors=apply_col, alpha=0.0007)
            file_name_mask=str(batch)+'_gt.png'
            imageio.imwrite(os.path.join(out,file_name_pred),(img_pred))
            imageio.imwrite(os.path.join(out,file_name_mask),(img_mask))
        else:
            img_pred=color.label2rgb(preds[i,:,:],original_image[i,:,:,:],colors=apply_col, alpha=0.0007)
            file_name_pred=str(batch)+'.png'
            imageio.imwrite(os.path.join(out,file_name_pred),(img_pred))

def eval_net(net, loader,criterion,criterion_dice,device):
    '''Forward passs with no grad'''
    
    net.eval()
    total = 0
    correct_pixels =0
    epoch_loss=0
    dice=0
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs=  torch.squeeze(imgs.to(device=device, dtype=torch.float32))
        masks= torch.squeeze(masks.to(device=device, dtype=torch.long))
        with torch.no_grad():
            preds = net(imgs)#['out']
        loss = criterion(preds,masks)
        loss_dice = criterion_dice(preds,masks)
        loss=loss+loss_dice
        epoch_loss += loss.item()
        dice+=dice_score(preds,masks).detach().cpu().numpy()
        _, preds = torch.max(preds.data, 1)
        total += masks.nelement()
        correct_pixels += preds.eq(masks.data).sum().item()

    return 100 * (correct_pixels / total),epoch_loss, dice

def display_images(imageSet,mutiple=True):
    '''This method takes a List of image and shows them in a thumbnil fashion'''
    import matplotlib.pyplot ; 

    if mutiple:
        # Define the grid size of the viewing
        height=np.ceil(np.sqrt(imageSet.shape[2])).astype(int)
        width=np.ceil((imageSet.shape[2])/height +  1)

        #Define the size each individule figure in the grid
        fig=matplotlib.pyplot.figure(figsize=(15,15))

        #Go through all the files and put them in the figure
        for idx in range(imageSet.shape[0]):
            fig.add_subplot(height,width,idx+1)
            matplotlib.pyplot.axis('off')
            matplotlib.pyplot.imshow(imageSet[idx,:,:] , cmap='gray')
        matplotlib.pyplot.show()
    else:
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.imshow(imageSet, cmap='gray')
        matplotlib.pyplot.show()

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        
        return loss

def dice_score(input: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
      
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + 1e-6)
        return torch.mean(dice_score)

def one_hot(labels: torch.Tensor,
                num_classes: int,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                eps: Optional[float] = 1e-6) -> torch.Tensor:
        r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

        Args:
            labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                    where N is batch siz. Each value is an integer
                                    representing correct classification.
            num_classes (int): number of classes in labels.
            device (Optional[torch.device]): the desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see torch.set_default_tensor_type()). device will be the CPU for CPU
            tensor types and the current CUDA device for CUDA tensor types.
            dtype (Optional[torch.dtype]): the desired data type of returned
            tensor. Default: if None, infers data type from values.

        Returns:
            torch.Tensor: the labels in one hot tensor.

        Examples::
            >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
            >>> tgm.losses.one_hot(labels, num_classes=3)
            tensor([[[[1., 0.],
                    [0., 1.]],
                    [[0., 1.],
                    [0., 0.]],
                    [[0., 0.],
                    [1., 0.]]]]
        """
        if not torch.is_tensor(labels):
            raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                            .format(type(labels)))
        if not len(labels.shape) == 3:
            raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                            .format(labels.shape))
        if not labels.dtype == torch.int64:
            raise ValueError(
                "labels must be of the same dtype torch.int64. Got: {}" .format(
                    labels.dtype))
        if num_classes < 1:
            raise ValueError("The number of classes must be bigger than one."
                            " Got: {}".format(num_classes))
        batch_size, height, width = labels.shape
        one_hot = torch.zeros(batch_size, num_classes, height, width,
                            device=device, dtype=dtype)
        return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
# This transfotmation are adapted from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/augment/transforms.py
class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, random_state, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.2, **kwargs):
        self.random_state = random_state
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            alpha = self.random_state.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m