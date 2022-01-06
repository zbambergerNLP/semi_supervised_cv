# Semi Supervised Computer Vision

We note the following about our implementation of MoCo and MoCo 2:

1. In order to reproduce runs, one must reconfigure the logic related to wandb. 
2. `training_loop.py` contains the primary logic for MoCo pre-training. As in the MoCo paper, we construct a queue from
which we draw images for the contrastive learning task (matching a query to its corresponding key, and not to 
alternative images via contrastive loss). We found that with the default training parameters (i.e., those set by the 
flag definitions at the top of the file) convergence within 2 epochs. Specifically, shortly after the first epoch, 
accuracy begins to rise dramatically until it reaches 100% and remains there. 
3. The encoder used during both pre-training and fine-tuning is defined in `models.py`. Note that we added an additional
linear layer as suggested by the MoCo 2 paper. This helped us significantly in both training stages of the model. 
4. The logic for image augmentation (used during pre-training) can be found in `augment.py`. Here we reproduce the 
augmentations suggested by both MoCo and MoCO 2. Namely, we add the Gaussian Blur augmentation on top of the ones 
suggested by the original MoCo paper. 
5. Our logic for fine-tuning the MoCo encoder to Imagenette is found in `fine_tuning.py`. After running the 
'training_loop' pre-training code mentioned above (with default parameters), we fine-tuned our model until convergence
at 2 epochs. Our training accuracy was ~92%, while the validation accuracy was ~89% (indicating relatively little 
over-fitting).
6. It is worth noting that we keep our constants in a separate file -- `consts.py`. This removes clutter from the other 
files and increases readability.
7. We also add two sweep yaml files that are used for wandb sweeps. However, it might be an issue to play with this
given point (1) above. 

## Dependencies:
* **pytorch**: conda install pytorch torchvision torchaudio -c pytorch
* **fastai**: conda install -c fastchan fastai anaconda
* **scikit image**: conda install scikit-image
* **wandb**: conda install -c conda-forge wandb
