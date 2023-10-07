# A human parsing and non-local feature extraction network for occluded person re-identification
Codes for 'A human parsing and non-local feature extraction network for occluded person re-identification'. 

Our model is based on the open object ,[ISP](https://github.com/CASIA-IVA-Lab/ISP-reID). Our code has made the following changes:

- We use a strategy to generate diverse occluded images which ISP doesn't have
- We use a pre-trained CE2P network instead of cluster which is used in ISP to generate pseudo-labels
- We add a new non-local block which aggregates similar features after the backbone
- We change the way to measure distance between features in our measurement strategy
## Results (rank1/mAP)
| Model | Market1501 |DukeMTMC-reID | CUHK03-D | CUHK03-L | Occluded-REID | Occluded-DukeMTMC |
| ----- | :--------: | :----------: | :------: | :------: | :-----------: | :---------------: |
|Ours   |0.954(0.893)|0.899(0.804)  |0.759(0.705)|0.784(0.737)|0.869      |0.672(0.564)|
## Install dependencies:

* pytorch>=1.1.0
* torchvision
* ignite
* yacs