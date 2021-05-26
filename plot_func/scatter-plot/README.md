## scatter plot画图程序说明

2021年4月24日 第一版



#### 用途介绍

本画图程序用于展示模型预测结果之间的对比，同时给出回归直线斜率、R方等信息。

<img src="/assets/b.jpg" alt="示意图1" style="zoom:20%;" />





#### 使用方法

参照/src/main.py文件内容。

```python
from scatter_plot import *
scatter_plot(A,B,namelist,savedir,maxcol=3,cb=True)
```

其中A，B 分别表示被画的数据，形状均为(m,n)，n为物种数、m为该物种的样本数；

namelist是一个列表，存储对应物种的名称；

maxcol是输出图像的列数；cb=True表示需要画出colorbar。

建议列数接近总子图数的平方根，这样的图像一般比较美观。

如果列数与行数差别过大，图像一些内部结构参数（如间距、字号、文字位置等等）需要人工调整。