## 环境
- pytorch 1.13.0
- cuda 11.6
- numpy
- transformers(使用预训练模型“roberta-base”)[预训练模型](https://huggingface.co/roberta-base)
- torch_geometric
- torch-scatter
-  torch-sparse

数据集下载地址：(http://data.sklccc.com/detail?id=40281aaf84852ff80184bd7f255138ba)

## 文件结构
- data文件夹里存放的是数据集
- dataset.py：生成数据的处理类；在本程序中，使用预训练模型提取Tweet、description、screen name等特征；
  - `splitData()`:读取数据集，切分、打乱数据
  - `dataSetGraph()`:数据处理类，其中使用预训练模型提取特征
- model.py：模型文件
- util.py:生成数据，加载数据
  - `makeGraphData()`:生成预处理的图数据
  - `getGraphData()`:从文件中读取预处理的数据
- main.py:程序入口，其中eval()函数获得测试集结果

## 处理流程
- 首先调用`makeGraphData(device, MAX_LEN)`获得预处理的数据。（已提前生成好）
- 调用`getGraphData()`读取存储在磁盘的预处理数据。
- 然后分别获得train、valid、test数据的数量。
- 加载`RobotGraphClassify()`模型获得测试集标签存储到`submission.csv`中。

## 特征提取

在本程序中，提取用户的"screen_name","description","tweet","followers_count","friends_count","listed_count","favourites_count","statuses_count","protected","geo_enabled","verified","contributors_enabled","is_translator","is_translation_enabled","default_profile_image"等特征。

其中，"screen_name","description","tweet"等信息使用预训练模型"roberta-base"进行特征提取。
"followers_count","friends_count","listed_count","favourites_count","statuses_count"等信息与账户的活跃程度相关，他们的数据形成profile向量。

"protected","geo_enabled","verified","contributors_enabled","is_translator","is_translation_enabled","default_profile_image"等信息在某种程度上是官方给予该账户的一些认证，如"protected"、"verified"等；以及一些反应账户活跃程度的相关信息，由于他们的取值皆为0和1，因此上述信息共同构成personal向量。

## 模型介绍

模型主要是图卷积神经网络，因为在社交网络，人与人之间是有一定的联系的，比如人们会关注和自己具有相同兴趣爱好的博主，具有相同兴趣爱好的人往往所发推文也是同一话题、同一情感的；因此使用图神经网络时，每个人的特征都不是孤立地，彼此之间是可以相互影响的，从他所关注的人以及被他人关注的账号中抽取信息，共同构建当前用户的特征向量。

### GCN简要

图卷积算子：
$$
h_i^{l+1} = \sigma(\sum_{j\in N_i}\frac{1}{c_{ij}}h_j^lw_{R_j}^l)
$$
其中，$h_i^l$为节点
在第l层的特征表达；$c_{ij}$归一化因子；$N_i$节点i的邻居；$R_i$节点i的类型；$w_{R_i}$表示$R_i$类型节点的变换权重参数。

图卷积的过程大致可以分为三步：

- 广播：每一个节点将自身的特征信息经过变换后发送给邻居节点，即在同一个网络中，相互连接的节点彼此共享特征信息。
- 接收：每个节点将邻居节点的特征信息聚集起来。
- 变换：将收到的信息做非线性变换，并构建出节点的特征信息。

### 模型架构
![img](./img/nlp_今年想去春茧咧.jpg)

### 调参

在多次实验中，模型对训练集中正负比比较敏感，一开始是随机打乱，效果并不理想,发现随机打乱时训练集的正负比会出现很大的波动，后来改为取一部分负样本、正样本后再打乱，二者比例从2：1到1：1之间逐步调整，找到了比较理想的切分。

在模型训练到后面时，适当降低学习率和Dropout，逐步调整，不过对提升F值效果甚微，主要还要是合理设置训练集。
