# BFENNet
This repository contains the Implementation details of the paper "Bio-inspired feature enhancement network for edge detection".
The address of the paper is at (https://link.springer.com/article/10.1007/s10489-022-03202-2)

## Citations
If you are using the code/model/data provided here in a publication, please consider citing our paper:
```
@article{lin2022bio,
  title={Bio-inspired feature enhancement network for edge detection},
  author={Lin, Chuan and Zhang, Zhenguang and Hu, Yihua},
  journal={Applied Intelligence},
  pages={1--16},
  year={2022},
  publisher={Springer}
}
```
## Get Start
1、Download our code.<br/>
2、prepare the dataset.<br/>
3、Configure the environment.<br/>
4、If Windows system, please modify the dataset in cfgs.yaml.<br/>
5、Run the "train.py".<br/>

文件夹中包含模型文件（model.py）
训练用文件（train.py）
测试用文件(test,py)
以及其他相关文件(cfgs...)

The folder contains model files (model.py)<br/>
training files (train.py)<br/>
testing files (test.py) and other related files (cfgs ...). 

BSDS_test和NYUD_test文件夹中包含本文模型在BSDS500数据集合NYUD数据集上的单尺度、多尺度测试结果。
The folders of BSDS_test and NYUD_test contain the single-scale and multi-scale test results of this model on NYUD data set of BSDS500 yuan data set.


## Datsets
We use the links in [RCF](https://github.com/yun-liu/rcf) Repository (really thanks for that).
The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:<br/>
```
  wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
  wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
  wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
```
# Reference and Acknowledgments
When building our codeWe referenced the repositories as follow:<br/>
1.[DRC](https://github.com/cyj5030/DRC-Release)
2.[RCF](https://github.com/yun-liu/rcf)
3.[HED](https://github.com/xwjabc/hed)
4.[DPED](https://github.com/cimerainbow/DPED)