# Graph-Laplacian-new
Graph Laplacian Regularization for Super-Resolution Task (Tensorflow)  
sample of paper:[New_Graph_Laplacian .pdf](https://github.com/p0367/Graph-Laplacian-new/files/6345015/New_Graph_Laplacian.pdf)
the code from my research  

## method
![幻灯片21 - 副本](https://user-images.githubusercontent.com/56641346/110503874-fba8d280-813f-11eb-8714-efbeb4e48f42.PNG)

## Model Architecture(example of SRCNN)
![image3](https://user-images.githubusercontent.com/56641346/110503975-12e7c000-8140-11eb-9f41-68964ff03609.png)

## Result
![report22](https://user-images.githubusercontent.com/56641346/110504172-3dd21400-8140-11eb-85b4-c7545d6be4c9.png)

- Manga109 Average Result
 
The average PSNR(dB) results of different methods on the Manga109 dataset.

|  Bicubic | SRCNN | **SRCNN+Laplacian(ours)**  | EDSR | **EDSR+Laplacian(ours)** |  WDSR |  **WDSR+Laplacian(ours)**  |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| 19.95  |  20.07  |   **21.00**   | 20.54  | **21.89** |  20.31 |  **21.84** |

## Dataset
I used the Manga109/Cartoon set dataset for training, you can download it here:   
Manga109:  
http://www.manga109.org/ja/index.html  
Cartoon set:  
https://google.github.io/cartoonset/

## Requirements
```
tensorflow >= 2
numpy
opencv
```

## Train
```
$ python main.py
```

## Note1
please unzip from .7z to get the sample dataset

## Note2
1. to change the model by changing the import,for example  
import model_EDSR → import model_WDSR  

2. can also change the scale in every model.py
