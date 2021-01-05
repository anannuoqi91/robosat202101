"""
geojson数据格式转换（墨卡托）
"""
# import argparse
# import csv
# import json
#
# from supermercado import burntiles
# from tqdm import tqdm
#
# with open(r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_test.json') as f:
#     features = json.load(f)
#
#     tiles = []
#
#     for feature in tqdm(features["features"], ascii=True, unit="feature"):
#         bound = burntiles.burn([feature], 18)
#         if bound is not False:
#             bound = bound.tolist()
#             tiles.extend(map(tuple, bound))
#
#     # tiles can overlap for multiple features; unique tile ids
#     tiles = list(set(tiles))
#
#     with open(r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_test.tiles', "w") as fp:
#         writer = csv.writer(fp)
#         writer.writerows(tiles)

"""
影像下载
"""
# import os
# import sys
# import time
# import argparse
# import concurrent.futures as futures
#
# import requests
# from PIL import Image
# from tqdm import tqdm
#
# from robosat.tiles import tiles_from_csv, fetch_image
#
#
# tiles = list(tiles_from_csv(r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_test.tiles'))
# out = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_test'
# iurl = r' http://58.210.9.134:8077/vt_all_new_one_map/data/sip_20181028_image_level_18/{z}/{x}/{y}.png'
# num_workers = 10
# rate = 10
# with requests.Session() as session:
#     # tqdm has problems with concurrent.futures.ThreadPoolExecutor; explicitly call `.update`
#     # https://github.com/tqdm/tqdm/issues/97
#     progress = tqdm(total=len(tiles), ascii=True, unit="image")
#
#     with futures.ThreadPoolExecutor(num_workers) as executor:
#         def worker(tile):
#             tick = time.monotonic()
#             x, y, z = map(str, [tile.x, tile.y, tile.z])
#             os.makedirs(os.path.join(out, z, x), exist_ok=True)
#             path = os.path.join(out, z, x, "{}.{}".format(y, 'webp'))
#             if os.path.isfile(path):
#                 return tile, True
#             url = iurl.format(x=tile.x, y=tile.y, z=tile.z)
#             res = fetch_image(session, url)
#             if not res:
#                 return tile, False
#             try:
#                 image = Image.open(res)
#                 image.save(path, optimize=True)
#             except OSError:
#                 return tile, False
#             tock = time.monotonic()
#             time_for_req = tock - tick
#             time_per_worker = num_workers / rate
#             if time_for_req < time_per_worker:
#                 time.sleep(time_per_worker - time_for_req)
#             progress.update()
#             return tile, True
#
#         for tile, ok in executor.map(worker, tiles):
#             if not ok:
#                 print("Warning: {} failed, skipping".format(tile), file=sys.stderr)

"""
掩膜
"""
# import argparse
# import collections
# import json
# import os
# import sys
#
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
#
# import mercantile
# from rasterio.crs import CRS
# from rasterio.transform import from_bounds
# from rasterio.features import rasterize
# from rasterio.warp import transform
# from supermercado import burntiles
#
# from robosat.config import load_config
# from robosat.colors import make_palette
# from robosat.tiles import tiles_from_csv
#
# def feature_to_mercator(feature):
#     """Normalize feature and converts coords to 3857.
#
#     Args:
#       feature: geojson feature to convert to mercator geometry.
#     """
#     # Ref: https://gist.github.com/dnomadb/5cbc116aacc352c7126e779c29ab7abe
#
#     src_crs = CRS.from_epsg(4326)
#     dst_crs = CRS.from_epsg(3857)
#
#     geometry = feature["geometry"]
#     if geometry["type"] == "Polygon":
#         xys = (zip(*part) for part in geometry["coordinates"])
#         xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
#
#         yield {"coordinates": list(xys), "type": "Polygon"}
#
#     elif geometry["type"] == "MultiPolygon":
#         for component in geometry["coordinates"]:
#             xys = (zip(*part) for part in component)
#             xys = (list(zip(*transform(src_crs, dst_crs, *xy))) for xy in xys)
#
#             yield {"coordinates": list(xys), "type": "Polygon"}
#
#
# def burn(tile, features, size):
#     """Burn tile with features.
#
#     Args:
#       tile: the mercantile tile to burn.
#       features: the geojson features to burn.
#       size: the size of burned image.
#
#     Returns:
#       image: rasterized file of size with features burned.
#     """
#
#     # the value you want in the output raster where a shape exists
#     burnval = 1
#     shapes = ((geometry, burnval) for feature in features for geometry in feature_to_mercator(feature))
#
#     bounds = mercantile.xy_bounds(tile)
#     transform = from_bounds(*bounds, size, size)
#
#     return rasterize(shapes, out_shape=(size, size), transform=transform)
#
#
# args_tiles = r'/Users/zhangqi/Documents/GitHub/robosat/data/H51H032022.tiles'
# args_size = 256
# args_zoom = 18
# args_dataset = r'/Users/zhangqi/Documents/GitHub/robosat/data/dataset-building-predict.toml'
# args_out = r'/Users/zhangqi/Documents/GitHub/robosat/data/masks'
# args_features = r'/Users/zhangqi/Documents/GitHub/robosat/data/H51H032022.geojson'
#
# dataset = load_config(args_dataset)
#
# classes = dataset["common"]["classes"]
# colors = dataset["common"]["colors"]
# assert len(classes) == len(colors), "classes and colors coincide"
#
# assert len(colors) == 2, "only binary models supported right now"
# bg = colors[0]
# fg = colors[1]
#
# os.makedirs(args_out, exist_ok=True)
#
# # We can only rasterize all tiles at a single zoom.
# assert all(tile.z == args_zoom for tile in tiles_from_csv(args_tiles))
#
# with open(args_features) as f:
#     fc = json.load(f)
#
# # Find all tiles the features cover and make a map object for quick lookup.
# feature_map = collections.defaultdict(list)
# for i, feature in enumerate(tqdm(fc["features"], ascii=True, unit="feature")):
#
#     if (feature["geometry"]["type"] != "Polygon") and (feature["geometry"]["type"] != "MultiPolygon"):
#         continue
#
#     try:
#         for tile in burntiles.burn([feature], zoom=args_zoom):
#             feature_map[mercantile.Tile(*tile)].append(feature)
#     except ValueError as e:
#         print("Warning: invalid feature {}, skipping".format(i), file=sys.stderr)
#         continue
#
# # Burn features to tiles and write to a slippy map directory.
# for tile in tqdm(list(tiles_from_csv(args_tiles)), ascii=True, unit="tile"):
#     if tile in feature_map:
#         out = burn(tile, feature_map[tile], args_size)
#     else:
#         out = np.zeros(shape=(args_size, args_size), dtype=np.uint8)
#
#     out_dir = os.path.join(args_out, str(tile.z), str(tile.x))
#     os.makedirs(out_dir, exist_ok=True)
#
#     out_path = os.path.join(out_dir, "{}.png".format(tile.y))
#
#     if os.path.exists(out_path):
#         prev = np.array(Image.open(out_path))
#         out = np.maximum(out, prev)
#
#     out = Image.fromarray(out, mode="P")
#
#     palette = make_palette(bg, fg)
#     out.putpalette(palette)
#
#     out.save(out_path, optimize=True)

"""
数据集分割
训练a、验证b、评估c
比例 a+b+c=1
内容不重复
"""
# import random
# import csv
# from robosat.tiles import tiles_from_csv
# from tqdm import tqdm
# import os
#
# dic = {'a':0, 'b':1, 'c':2}
# for i in tqdm(dic, desc="split", ascii=True):
#     print(i, dic[i])
#
# path = r'/Users/zhangqi/Documents/GitHub/robosat/data/H51H032022.tiles'
# out = r'/Users/zhangqi/Documents/GitHub/robosat/data'
# n_training = 0.8
# n_validation = 0.1
# n_evaluation = 0.1
#
#
# def difference_set(x, y):
#     """
#     x包含y(set)
#     :param x:
#     :param y:
#     :return:
#     """
#     if y is not None:
#         return x - y, None
#     else:
#         return x, None
#
# label = ['training', 'validation', 'evaluation']
# data_set = {'training': None,
#             'validation': None,
#             'evaluation': None
#             }
# data_rate = {'training': n_training,
#              'validation': n_validation,
#              'evaluation': n_evaluation
#              }
#
# tiles = set(tiles_from_csv(path))
# num = len(tiles)
# for data_label in tqdm(label, desc="split", ascii=True):
#     tmp_out = os.path.join(out, 'csv_' + data_label + '.tiles')
#     rate = data_rate[data_label]
#     for i in data_set.keys():
#         if i != data_label:
#             tiles, data_set[i] = difference_set(tiles, data_set[i])
#     tiles_list = list(tiles)
#     tiles_out = random.sample(tiles_list, int(num * rate))
#     with open(tmp_out, "w") as fp:
#         writer = csv.writer(fp)
#         writer.writerows(tiles_out)

"""
分配训练数据、验证数据、评估数据
"""
# import os
# import argparse
# import shutil
#
# from tqdm import tqdm
#
# from robosat.tiles import tiles_from_slippy_map, tiles_from_csv
#
#
# args_images = r'/Users/zhangqi/Documents/GitHub/robosat/data/masks/'
# args_tiles = r'/Users/zhangqi/Documents/GitHub/robosat/data/csv_training.tiles'
# args_out = r'/Users/zhangqi/Documents/GitHub/robosat/data/dataset/training/labels'
#
# images = tiles_from_slippy_map(args_images)
#
# tiles = set(tiles_from_csv(args_tiles))
#
# for tile, src in tqdm(list(images), desc="Subset", unit="image", ascii=True):
#     if tile not in tiles:
#         continue
#
#     # The extention also includes the period.
#     extention = os.path.splitext(src)[1]
#
#     os.makedirs(os.path.join(args_out, str(tile.z), str(tile.x)), exist_ok=True)
#     dst = os.path.join(args_out, str(tile.z), str(tile.x), "{}{}".format(tile.y, extention))
#
#     shutil.copyfile(src, dst)

"""
权重计算
"""
# import os
# import argparse
#
# import numpy as np
# from tqdm import tqdm
#
# import torch
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose
#
# from robosat.config import load_config
# from robosat.datasets import SlippyMapTiles
# from robosat.transforms import ConvertImageMode, MaskToTensor
#
# args_dataset = r'/Users/zhangqi/Documents/GitHub/robosat/data/dataset-building-weights.toml'
# dataset = load_config(args_dataset)
#
# path = dataset["common"]["dataset"]
# num_classes = len(dataset["common"]["classes"])
#
# train_transform = Compose([ConvertImageMode(mode="P"), MaskToTensor()])
#
# train_dataset = SlippyMapTiles(os.path.join(path, "training", "labels"), transform=train_transform)
#
# n = 0
# counts = np.zeros(num_classes, dtype=np.int64)
#
# loader = DataLoader(train_dataset, batch_size=1)
# for images, tile in tqdm(loader, desc="Loading", unit="image", ascii=True):
#     image = torch.squeeze(images)
#
#     image = np.array(image, dtype=np.uint8)
#     n += image.shape[0] * image.shape[1]
#     counts += np.bincount(image.ravel(), minlength=num_classes)
#
# assert n > 0, "dataset with masks must not be empty"
#
# # Class weighting scheme `w = 1 / ln(c + p)` see:
# # - https://arxiv.org/abs/1707.03718
# #     LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
# # - https://arxiv.org/abs/1606.02147
# #     ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
#
# probs = counts / n
# weights = 1 / np.log(1.02 + probs)
#
# weights.round(6, out=weights)
# print(weights.tolist())

"""
训练
"""

"""
预测建筑概率
"""
# import argparse
# import os
# import sys
#
# import numpy as np
#
# import torch
# import torch.nn as nn
# import torch.backends.cudnn
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, Normalize
#
# from tqdm import tqdm
# from PIL import Image
#
# from robosat.datasets import BufferedSlippyMapDirectory
# from robosat.unet import UNet
# from robosat.config import load_config
# from robosat.colors import continuous_palette_for_color
# from robosat.transforms import ConvertImageMode, ImageToTensor
#
#
# args_model = r'/Users/zhangqi/Documents/GitHub/robosat/data/model-unet.toml'
# args_dataset = r'/Users/zhangqi/Documents/GitHub/robosat/data/dataset-building-predict.toml'
# args_checkpoint = r'/Users/zhangqi/Documents/GitHub/robosat/data/checkpoint/checkpoint-00008-of-00010.pth'
# args_tiles = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_test'
# args_tile_size = 256
# args_overlap = 32
# args_batch_size = 2
# args_workers = 0
# args_probs = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_segmentation-probabilities'
#
# model = load_config(args_model)
# dataset = load_config(args_dataset)
#
# cuda = model["common"]["cuda"]
#
# device = torch.device("cuda" if cuda else "cpu")
#
# def map_location(storage, _):
#     return storage.cuda() if cuda else storage.cpu()
#
# if cuda and not torch.cuda.is_available():
#     sys.exit("Error: CUDA requested but not available")
#
# num_classes = len(dataset["common"]["classes"])
#
# # https://github.com/pytorch/pytorch/issues/7178
# chkpt = torch.load(args_checkpoint, map_location=map_location)
#
# net = UNet(num_classes).to(device)
# net = nn.DataParallel(net)
#
# if cuda:
#     torch.backends.cudnn.benchmark = True
#
# net.load_state_dict(chkpt["state_dict"])
# net.eval()
#
# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#
# transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])
#
# directory = BufferedSlippyMapDirectory(args_tiles, transform=transform, size=args_tile_size, overlap=args_overlap)
# assert len(directory) > 0, "at least one tile in dataset"
#
# loader = DataLoader(directory, batch_size=args_batch_size, num_workers=args_workers)
#
# # don't track tensors with autograd during prediction
# with torch.no_grad():
#     for images, tiles in tqdm(loader, desc="Eval", unit="batch", ascii=True):
#         images = images.to(device)
#         outputs = net(images)
#
#         # manually compute segmentation mask class probabilities per pixel
#         probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()
#
#         for tile, prob in zip(tiles, probs):
#             x, y, z = list(map(int, tile))
#
#             # we predicted on buffered tiles; now get back probs for original image
#             prob = directory.unbuffer(prob)
#
#             # Quantize the floating point probabilities in [0,1] to [0,255] and store
#             # a single-channel `.png` file with a continuous color palette attached.
#
#             assert prob.shape[0] == 2, "single channel requires binary model"
#             assert np.allclose(np.sum(prob, axis=0), 1.), "single channel requires probabilities to sum up to one"
#             foreground = prob[1:, :, :]
#
#             anchors = np.linspace(0, 1, 256)
#             quantized = np.digitize(foreground, anchors).astype(np.uint8)
#
#             palette = continuous_palette_for_color("pink", 256)
#
#             out = Image.fromarray(quantized.squeeze(), mode="P")
#             out.putpalette(palette)
#
#             os.makedirs(os.path.join(args_probs, str(z), str(x)), exist_ok=True)
#             path = os.path.join(args_probs, str(z), str(x), str(y) + ".png")
#
#             out.save(path, optimize=True)

"""
预测概率转换为建筑物掩模
"""

# import os
# import sys
# import argparse
#
# import numpy as np
#
# from tqdm import tqdm
# from PIL import Image
#
# from robosat.tiles import tiles_from_slippy_map
# from robosat.colors import make_palette
#
# def softvote(probs, axis=0, weights=None):
#     """Weighted average soft-voting to transform class probabilities into class indices.
#
#     Args:
#       probs: array-like probabilities to average.
#       axis: axis or axes along which to soft-vote.
#       weights: array-like for weighting probabilities.
#
#     Notes:
#       See http://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting
#     """
#
#     return np.argmax(np.average(probs, axis=axis, weights=weights), axis=axis)
#
# args_weights = None
# args_probs = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_segmentation-probabilities'
# args_masks = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_segmentation-masks'
#
# if args_weights and len(args_probs) != len(args_weights):
#     sys.exit("Error: number of slippy map directories and weights must be the same")
#
# tilesets = map(tiles_from_slippy_map, [args_probs])
#
# for tileset in tqdm(list(zip(*tilesets)), desc="Masks", unit="tile", ascii=True):
#     tiles = [tile for tile, _ in tileset]
#     paths = [path for _, path in tileset]
#
#     assert len(set(tiles)), "tilesets in sync"
#     x, y, z = tiles[0]
#
#     # Un-quantize the probabilities in [0,255] to floating point values in [0,1]
#     anchors = np.linspace(0, 1, 256)
#
#     def load(path):
#         # Note: assumes binary case and probability sums up to one.
#         # Needs to be in sync with how we store them in prediction.
#
#         quantized = np.array(Image.open(path).convert("P"))
#
#         # (512, 512, 1) -> (1, 512, 512)
#         foreground = np.rollaxis(np.expand_dims(anchors[quantized], axis=0), axis=0)
#         background = np.rollaxis(1. - foreground, axis=0)
#
#         # (1, 512, 512) + (1, 512, 512) -> (2, 512, 512)
#         return np.concatenate((background, foreground), axis=0)
#
#     probs = [load(path) for path in paths]
#
#     mask = softvote(probs, axis=0, weights=args_weights)
#     mask = mask.astype(np.uint8)
#
#     palette = make_palette("denim", "orange")
#     out = Image.fromarray(mask, mode="P")
#     out.putpalette(palette)
#
#     os.makedirs(os.path.join(args_masks, str(z), str(x)), exist_ok=True)
#
#     path = os.path.join(args_masks, str(z), str(x), str(y) + ".png")
#     out.save(path, optimize=True)

"""
建筑物掩模转换为 geojson
"""
import argparse

import numpy as np

from PIL import Image
from tqdm import tqdm

from robosat.tiles import tiles_from_slippy_map
from robosat.config import load_config

from robosat.features.parking import ParkingHandler


# Register post-processing handlers here; they need to support a `apply(tile, mask)` function
# for handling one mask and a `save(path)` function for GeoJSON serialization to a file.
handlers = {"parking": ParkingHandler}

args_dataset = r'/Users/zhangqi/Documents/GitHub/robosat/data/dataset-building-predict.toml'
args_masks = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_segmentation-masks'
args_type = 'parking'
args_out = r'/Users/zhangqi/Documents/GitHub/robosat/data/predict_geojson_features'

dataset = load_config(args_dataset)

labels = dataset["common"]["classes"]
assert set(labels).issuperset(set(handlers.keys())), "handlers have a class label"
index = labels.index(args_type)

handler = handlers[args_type]()

tiles = list(tiles_from_slippy_map(args_masks))

for tile, path in tqdm(tiles, ascii=True, unit="mask"):
    image = np.array(Image.open(path).convert("P"), dtype=np.uint8)
    mask = (image == index).astype(np.uint8)

    handler.apply(tile, mask)

handler.save(args_out)