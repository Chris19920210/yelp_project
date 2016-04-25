import mxnet as mx
import numpy as np
import os
import argparse

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='load pretrained model, results will be save in current dir')
parser.add_argument('--data-dir', type=str, default='data',
                    help='the input data directory')
parser.add_argument('--model-prefix', type=str, default='Inception_BN',
                    help='the prefix of the model to load')
parser.add_argument('--load-epoch', type=int, required=True,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--dataset', type=str, required=True,
                    help='train dataset name')
parser.add_argument('--data-shape', type=int, default=224,
                    help='set image\'s shape')
args = parser.parse_args()

# network

# data iterator
data_shape = (3, args.data_shape, args.data_shape)
dataiter = mx.io.ImageRecordIter(
    path_imgrec=os.path.join(args.data_dir, args.dataset),
    mean_r=117,
    mean_g=117,
    mean_b=117,
    data_shape=data_shape,
    batch_size=args.batch_size
)
# Load the pre-trained model
prefix = "model" + "/" + args.model_prefix
num_round = args.load_epoch
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu())
# get internals from model's symbol
internals = model.symbol.get_internals()
# get feature layer symbol out of internals
fea_symbol = internals["global_pool_output"]
# Feedforward model get features
feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=128, arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
global_pooling_feature = feature_extractor.predict(dataiter)

a, b, c, d = global_pooling_feature.shape
value = global_pooling_feature.reshape(a, b)

np.save("./" + args.model_prefix + '_' + args.dataset[:args.dataset.find('.')] + '.npy', value.astype(np.float32))

