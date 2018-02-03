# -*- coding: utf-8 -*-

import gzip
import copy
import numpy as np
from PIL import Image
import paddle.v2 as paddle

DATA_DIM=224*224*3
DICT_DIM=10000
image = paddle.layer.data(name="image", type=paddle.data_type.dense_vector(DATA_DIM))
target = paddle.layer.data(name="target", type=paddle.data_type.integer_value_sequence(DICT_DIM))
label = paddle.layer.data(name="label", type=paddle.data_type.integer_value_sequence(DICT_DIM))
paddle.init(use_gpu=False, trainer_count=1)

