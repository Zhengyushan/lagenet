## LAGE-Net

This is a PyTorch implementation of the paper [Zheng et al. Encoding histopathology WSIs with location-aware graphs for diagnostically relevant regions retrieval, Medical Image Analysis, 2022](https://doi.org/10.1016/j.media.2021.102308):
```
@Article{zheng2022encoding,
  author  = {Zheng, Yushan and Jiang, Zhiguo and Shi, Jun and Xie, Fengying and Zhang, Haopeng and 
             Luo, Wei and Hu, Dingyi and Sun, Shujiao and Jiang, Zhongmin and Xue, Chenghai},
  title   = {Encoding histopathology whole slide images with location-aware graphs for diagnostically relevant regions retrieval},
  journal = {Medical Image Analysis},
  year    = {2022},
  volumn  = {76},
  pages   = {102308},
  doi     = {https://doi.org/10.1016/j.media.2021.102308},
}
```

### Training
To train the LAGE-Net, please refer to  [run.sh](./run.sh):

### Data description
The structure of the whole slide image dataset to run the code.
```
/media/dataset/endometrial                # The directory of the data.
├─ 0A00DD22-A08E-4B47-A51B-94A8BD039DAA   # The directory for a slide, which is named by GUID in our dataset.
│  ├─ Large                               # The directory of image tiles in Level 0 (40X lens).
│  │  ├─ 0000_0000.jpg                    # The image tile in Row 0 and Column 0.
│  │  ├─ 0000_0001.jpg                    # The image tile in Row 0 and Column 1.
│  │  └─ ...
│  ├─ Medium                              # The directory of image tiles in Level 1 (20X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Small                               # The directory of image tiles in Level 2 (10X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Overview                            # The directory of image tiles in Level 3 (5X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Overview.jpg                        # The thumbnail of the WSI in Level 3.          
│  └─ AnnotationMask.png                  # The pixel-wise annotation mask of the WSI in Level 3.
├─ 0A003711-3BE4-44E2-9280-89D84E5AF59F
└─ ...
```