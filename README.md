# robosat202101
robosat源码解析及部分修改（截止当前测试，针对building的矢量化代码缺少；源码中针对parking的矢量化代码存在问题，截止到“预测概率转换为建筑物掩模”的代码可用）


- [RoboSat 自动提取建筑物]
    - [1. 数据准备工作](#1-数据准备工作)
        - [1.1 建筑物轮廓矢量数据](#11-建筑物轮廓矢量数据)
        - [1.2 提取训练区覆盖的瓦片行列号](#12-提取训练区覆盖的瓦片行列号)
        - [1.3 下载训练区遥感影像瓦片](#13-下载训练区遥感影像瓦片)
        - [1.4 制作训练区矢量数据蒙版标记](#14-制作训练区矢量数据蒙版标记)
    - [2. 训练和建模](#2-训练和建模)
        - [2.1 分配训练数据、验证数据、评估数据](#21-分配训练数据验证数据评估数据)
        - [2.2 权重计算](#22-权重计算)
        - [2.3 开始训练](#23-开始训练)
    - [3. 预测](#3-预测)
        - [3.1 准备预测区域数据](#31-准备预测区域数据)
        - [3.2 预测待提取建筑物概率](#32-预测待提取建筑物概率)
        - [3.3 预测概率转换为建筑物掩模](#33-预测概率转换为建筑物掩模)

## 1. 数据准备工作

### 1.1 建筑物轮廓矢量数据
​    已有的建筑物轮廓矢量数据用来作为建筑物提取的训练数据源。可以有两种方式获取：
- OSM 数据源，可以在 [geofabrik](http://download.geofabrik.de/) 获取，通过 [osmium](https://github.com/osmcode/osmium-tool) 和 [robosat](https://github.com/mapbox/robosat) 工具[进行处理](https://www.openstreetmap.org/user/daniel-j-h/diary/44321)。
- 自有数据源。通过 [QGIS](https://qgis.org/en/site/) 或 ArcMap 等工具，加载遥感影像底图，描述的建筑物轮廓 Shapefile 数据。

​    **本文使用第二种数据来源，并已[开源数据源](https://github.com/geocompass/robosat_buildings_training/tree/master/shp_data)，开源的矢量数据覆盖厦门核心区。**

### 1.2 提取训练区覆盖的瓦片行列号
​    使用 robosat 的 [cover](https://github.com/mapbox/robosat#rs-cover) 命令，即可获取当前训练区矢量数据覆盖的瓦片行列号，并使用 csv 文件存储。
```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu cover  --zoom 18 /data/buildings.json /data/buildings.tiles
```
   `cover`  命令的参数介绍：

> usage: `./rs cover [-h] --zoom ZOOM features out`
>
> - positional arguments:
>   - `features`     path to GeoJSON features
>   - `out`         path to csv file to store tiles in
> - optional arguments:
>   - `-h`, --help   show this help message and exit
>   - `--zoom ZOOM`  zoom level of tiles (default: None)

这里是获取在 18 级下训练区覆盖的瓦片行列号。18 级是国内地图通用的最大级别，如果有国外更清晰数据源，可设置更高地图级别（**目前解析源码过程中，在矢量化过程中未发现支持18级以外的数据**）。

​    cover 工具对训练区矢量数据计算的瓦片行列号使用的是通用的 WGS84->Web 墨卡托投影坐标系。

### 1.3 下载训练区遥感影像瓦片

​    使用 robosat 的 [download](https://github.com/mapbox/robosat#rs-download) 工具，即可获取当前训练区矢量数据覆盖的遥感影像，下载的瓦片通过 2.3 节中的**buildings.tiles** 确定。

```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu download http://ditu.google.cn/maps/vt/lyrs=s&x={x}&y={y}&z={z} /data/buildings.tiles /data/tiles
```

​    `download` 命令的参数介绍：

> usage: `./rs download [-h] [--ext EXT] [--rate RATE] url tiles out`
>
> - positional arguments:
>   - `url`  endpoint with {z}/{x}/{y} variables to fetch image tiles from
>   - `tiles`  path to .csv tiles file
>   - `out`  path to slippy map directory for storing tiles
> - optional arguments:
>   - `-h`, --help   show this help message and exit
>   - `--ext EXT` file format to save images in (default: webp)
>   - `--rate RATE` rate limit in max. requests per second (default: 10)

### 1.3 制作训练区矢量数据蒙版标记

​    使用 2.2 节中制作的 geojson 数据，通过 robosat 的 [rasterize](https://github.com/mapbox/robosat#rs-rasterize) 工具可制作训练区矢量数据的蒙版标记数据。蒙版标记数据与瓦片数据一一相对应，使用同样的 **buildings.tiles** 瓦片列表产生。

```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu rasterize --dataset /data/dataset-building.toml --zoom 18 --size 256 /data/buildings.json /data/buildings.tiles /data/masks
```

​    `rasterise` 命令的参数介绍：

> usage: `./rs rasterize [-h] --dataset DATASET --zoom ZOOM [--size SIZE] features tiles out`
>
> - positional arguments:
>   - `features` path to GeoJSON features file
>   - `tiles` path to .csv tiles file
>   - `out` directory to write converted images
>
> - optional arguments:
>     `-h`, `--help` show this help message and exit
>     `--dataset DATASET` path to dataset configuration file (default: None)
>     `--zoom ZOOM`  zoom level of tiles (default: None)
>     `--size SIZE` size of rasterized image tiles in pixels (default: 512)

  这里使用到了 `dataset-building.toml` 配置文件，文件中配置了瓦片地图路径、分类方式、蒙版标记的颜色等信息。示例配置可以查看官方示例文件 [dataset-parking.toml](https://github.com/mapbox/robosat/blob/master/config/dataset-parking.toml) ，配置内容如下：
  （**classes中的buildings在示例文件中是parking，该值只在矢量化过程中有影响，目前features代码提供parking的矢量化，但是矢量化代码存在一些bug有待修复**）

```toml
# Configuration related to a specific dataset.
# For syntax see: https://github.com/toml-lang/toml#table-of-contents

# Dataset specific common attributes.
[common]

  # The slippy map dataset's base directory.
  dataset = '/Users/wucan/Document/robosat/tiles/'

  # Human representation for classes. 
  classes = ['background', 'buildings']

  # Color map for visualization and representing classes in masks.
  # Note: available colors can be found in `robosat/colors.py`
  colors  = ['denim', 'orange']
```

​    配置文档中，最重要的是配置 `dataset` 目录，也就是上一步中下载的遥感影像瓦片路径。

​    至此，训练和建模所需的瓦片和蒙版标记已经全部准备完，分别在 `tiles` 和 `masks` 目录中。


## 2. 训练和建模

### 2.1 分配训练数据、验证数据、评估数据

​    RoboSat 分割模型是一个完全卷积的神经网络，需要将上一步准备好的数据集拆分为三部分，分别为`训练数据集` 、`验证数据集`、`评估数据集`，比例分别为80%、10%、10%。每一部分的数据集中，都包含影像瓦片和蒙版标记瓦片。

- 训练数据集：a training dataset on which we train the model on
- 验证数据集：a validation dataset on which we calculate metrics on after training
- 评估数据集：a hold-out evaluation dataset if you want to do hyper-parameter tuning

（**在robosat/tools 中新增split.py，用于进行数据分配**）

```shell
robosat202101
|  robosat
|  |  tools
|  |  |  split.py
```
​    `split` 命令的参数介绍：

> usage: `./rs download [-h] url tiles out`
>
> - positional arguments:
>   - `tiles`  csv to filter images by
>   - `training`  percentage of training data set
>   - `validation`  percentage of validation data set
>   - `evaluation`  percentage of evaluation data set
>   - `out`  directory to save filtered tiles to
> - optional arguments:
>   - `-h`, --help   show this help message and exit

   将步骤 1 中的数据进行随机分配的过程非常简单：

- 新建三个 csv 文件： `csv_training.tiles` 、`csv_validation.tiles`、 `csv_evaluation.tiles` 
- 将 `buildings.tiles` 中的瓦片列表随机按 80%、10%、10% 比例进行拷贝与粘贴。注意三个文件间的瓦片列表内容不能重复。

​    使用 RoboSat 中的 [subset](https://github.com/mapbox/robosat#rs-subset) 命令，将 `tiles` 和 `masks` 中的瓦片和蒙版按照上面三个 csv 文件的瓦片列表分配进行组织影像瓦片和蒙版标记数据。

```shell
# 准备训练数据集
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu subset /data/tiles/ /data/csv_training.tiles /data/dataset/training/images
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu subset /data/masks/ /data/csv_training.tiles /data/dataset/training/labels
# 准备验证数据集
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu subset /data/tiles/ /data/csv_validation.tiles /data/dataset/validation/images
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu subset /data/masks/ /data/csv_validation.tiles /data/dataset/validation/labels
# 准备评估数据集
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu subset /data/tiles/ /data/csv_evaluation.tiles /data/dataset/evaluation/images
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu subset /data/masks/ /data/csv_evaluation.tiles /data/dataset/evaluation/labels
```

​    `subset` 命令的参数介绍：

> usage: `./rs subset [-h] images tiles out`
>
> - positional arguments:
>   - `images` directory to read slippy map image tiles from for filtering
>   - ` tiles` csv to filter images by
>   - `out` directory to save filtered images to
>
> - optional arguments:
>   - `-h`, `--help`  show this help message and exit

​    分类完成以后，将会生成 `/data/dataset` 目录，目录结构如下：

```shell
dataset
|  training
|  |  images
|  |  labels
|  validataion
|  |  images
|  |  labels
|  evaluation
|  |  images
|  |  labels
```

### 2.2 权重计算

​    因为前景和背景在数据集中分布不均，可以使用 RoboSat 中的 [weights](https://github.com/mapbox/robosat#rs-weights) 命令，在模型训练之前计算一下每个类的分布。

```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu weights --dataset /data/dataset-building-weights.toml
```

​    `weights` 命令的参数如下：

> usage: `./rs weights [-h] --dataset DATASET`
>
> - optional arguments:
>   - `-h`, `--help`  show this help message and exit
>   - `--dataset DATASET` path to dataset configuration file (default: None)  

​    这里，用到了`dataset-building-weights.toml` ，是将前面步骤中的 `dataset-building.toml` 瓦片路径修改为包含训练数据集 `dataset` 的路径。执行权重计算命令后，得到权重为：`values = [XXX, XXX]` 。将其追加到 `dataset-building-weights.toml` 文件中，结果如下。

```
# Configuration related to a specific dataset.
# For syntax see: https://github.com/toml-lang/toml#table-of-contents


# Dataset specific common attributes.
[common]

  # The slippy map dataset's base directory.
  dataset = '/data/dataset'

  # Human representation for classes.
  classes = ['background', 'buildings']

  # Color map for visualization and representing classes in masks.
  # Note: available colors can be found in `robosat/colors.py`
  colors  = ['denim', 'orange']

# Dataset specific class weights computes on the training data.
# Needed by 'mIoU' and 'CrossEntropy' losses to deal with unbalanced classes.
# Note: use `./rs weights -h` to compute these for new datasets.
[weights]
  values = [XXX, XXX]
```

### 2.3 开始训练

​    RoboSat 使用 [train](https://github.com/mapbox/robosat#rs-train) 命令进行训练。

```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu train --model /data/model-unet.toml --dataset /data/dataset-building-weights.toml
```

​    `train` 命令的参数如下：

> usage: `./rs train [-h] --model MODEL --dataset DATASET [--checkpoint CHECKPOINT] [--resume RESUME] [--workers WORKERS]`
>
> - positional arguments:
>   - `--model MODEL` path to model configuration file (default: None)
>   - `--dataset DATASET` path to dataset configuration file (default: None)
>
> - optional arguments:
>   - `-h`, `--help show this help message and exit
>   - `--checkpoint CHECKPOINT`  path to a model checkpoint (to retrain) (default: None)
>   - `--resume RESUME`   resume training or fine-tuning (if checkpoint)  (default: False)
>   - `--workers WORKERS`  number of workers pre-processi ng images (default: 0)  

​    这里多了一个配置文件 `model-unet.toml` ，这个配置文件主要用来配置训练过程中的参数，包括是否启用 CUDA 、训练批次大小、影像瓦片的像素大小、检查点存储路径等。官方给出了[示例配置文件](https://github.com/mapbox/robosat/blob/master/config/model-unet.toml)，根据本实验的情况做了修改如下，配置如下。

```
# Configuration related to a specific model.
# For syntax see: https://github.com/toml-lang/toml#table-of-contents

# Model specific common attributes.
[common]

  # Use CUDA for GPU acceleration.
  cuda       = false

  # Batch size for training.
  batch_size = 2

  # Image side size in pixels.
  image_size = 256

  # Directory where to save checkpoints to during training.
  checkpoint = '/data/checkpoint/'


# Model specific optimization parameters.
[opt]

  # Total number of epochs to train for.
  epochs     = 10

  # Learning rate for the optimizer.
  lr         = 0.01

  # Loss function name (e.g 'Lovasz', 'mIoU' or 'CrossEntropy')
  loss = 'Lovasz'
```

​    RoboSat 会进行多次迭代训练，每次迭代训练都会保存检查点(checkpoint)和各项指标等。其中，训练日志例如：

```shell
--- Hyper Parameters on Dataset: /data/dataset ---
Batch Size:	 2
Image Size:	 256
Learning Rate:	 0.0001
Loss function:	 Lovasz
Weights :	 [1.644471, 5.409126]
---
Epoch: 1/10
Train    loss: 0.3190, mIoU: 0.410, buildings IoU: 0.017, MCC: -0.002
Validate loss: 0.3171, mIoU: 0.405, buildings IoU: 0.000, MCC: nan

...

Epoch: 10/10
Train    loss: 0.2693, mIoU: 0.528, buildings IoU: 0.229, MCC: 0.330
Validate loss: 0.2880, mIoU: 0.491, buildings IoU: 0.167, MCC: 0.262
```

​    可以选择最好的训练结果，保留其检查点( `checkpoint-***.pth` )，进入下一步 `predict`。一般来说，最后一个检查点效果最好。

## 3. 预测

### 3.1 准备预测区域数据

​    RoboSat 仅支持从影像瓦片中提取建筑物，不支持从任意的 jpg 图片中提取。所以我们需要先准备预测区域的瓦片数据。

​    通过 [geojson.io](http://geojson.io/) 绘制想要提取建筑物的范围，使用矩形框即可。将自动生成的 geojson 保存为 `predict_test.json`。

​    通过 1.2 中的 `cover` 命令，获取待提取范围的瓦片列表 csv 文件，保存到 `buildings_predict.tiles` 文件中。

```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu cover  --zoom 18 /data/shp_data/predict_test.json /data/buildings_predict.tiles
```

​    使用 1.3 中的 `download` 命令，下载待提取范围的影像瓦片，保存到 `images_predict` 文件夹中。

```shell
docker run -it --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu download https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x} /data/buildings_predict.tiles /data/images_predict
```


### 3.2 预测待提取建筑物概率

​    使用保存的检查点来（`checkpint`）预测图像中每个像素的分割概率，这些分割概率表示每个像素是建筑物还是背景的可能性，然后可以将这些概率转换为离散的分割掩模。

​    通过 RoboSat 的 [predict](https://github.com/mapbox/robosat#rs-predict) 命令，将待预测区域的建筑物（ `images_predict` ）提取为分割概率（`predict_segmentation-probabilities`）。

```shell
docker run -it -d --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu predict --tile_size 256 --model /data/model-unet.toml --dataset /data/dataset-building.toml --checkpoint /data/checkpoint/checkpoint-00010-of-00010.pth /data/images_predict /data/predict_segmentation-probabilities
```

​    `predict` 命令的参数如下：

> usage: `./rs predict [-h] [--batch_size BATCH_SIZE] --checkpoint CHECKPOINT [--overlap OVERLAP] --tile_size TILE_SIZE [--workers WORKERS] --model MODEL --dataset DATASET tiles probs`
>
> - positional arguments:
>   - `tiles`   directory to read slippy map image tiles from
>   - `probs`  directory to save slippy map probability masks to
>
> - optional arguments:
>   - `-h`, `--help  show this help message and exit
>      `
>   - `--batch_size BATCH_SIZE`  images per batch (default: 1)
>   - `--checkpoint CHECKPOINT`  model checkpoint to load (default: None)
>   - `--overlap OVERLAP` tile pixel overlap to predict on (default: 32)
>   - `--tile_size TILE_SIZE` tile size for slippy map tiles (default: None)
>   - `--workers WORKERS` number of workers pre-processing images (default: 0)
>   - `--model MODEL` path to model configuration file (default: None)
>   - `--dataset DATASET`  path to dataset configuration file (default: None)

### 3.3 预测概率转换为建筑物掩模

​    通过 RoboSat 的 [masks](https://github.com/mapbox/robosat#rs-masks) 命令，将上一步中的建筑物预测概率结果转换为建筑物掩模（`masks`），保存到 `predict_segmentation-masks` 文件夹中。

```shell
docker run -it -d --rm -v $PWD:/data --ipc=host --network=host mapbox/robosat:latest-cpu masks /data/predict_segmentation-masks /data/predict_segmentation-probabilities
```

> usage: `./rs masks [-h] [--weights WEIGHTS [WEIGHTS ...]] masks probs [probs ...]`
>
> - positional arguments:
>   - `masks`  slippy map directory to save masks to
>   - `probs`  slippy map directories with class probabilities
>
> - optional arguments:
>   -  `-h`, `--help` show this help message and exit
>   - `--weights WEIGHTS [WEIGHTS ...]`  weights for weighted average soft-voting (default:
>                             None)
