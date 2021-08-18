import argparse
import pickle

from load_data import Data
import random
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=500, help='Number of epochs (default: 200)')
parser.add_argument('--batchsize', type=int, default=128, help='Batch size (default: 128)')
parser.add_argument('--num_filters', type=int, default=9, help='number of filters CNN')
parser.add_argument('--filt_h', type=int, default=1, help='filt_h of  CNN')
parser.add_argument('--filt_w', type=int, default=2, help='filt_w of CNN')
parser.add_argument('--conv_stride', type=int, default=2, help='stride of CNN')
parser.add_argument('--h', type=int, default=1, help='h')
parser.add_argument('--w', type=int, default=600, help='w')
parser.add_argument('--pool_h', type=int, default=1, help='pool_h')
parser.add_argument('--pool_w', type=int, default=2, help='pool_w')
parser.add_argument('--pool_stride', type=int, default=2, help='pool_w')
parser.add_argument('--embsize_entity', default=600, help='Entity Embedding size (default: 200)')
parser.add_argument('--embsize_entity_type', default=200, help='Entity Type Embedding size (default: 100)')
parser.add_argument('--learningrate', default=0.0001, help='Learning rate (default: 0.00005)')
parser.add_argument("--droupt_input", default=0.0, type=float, help="droupt regularization item 0.2")
parser.add_argument("--droupt_feature", default=0.0, type=float, help="droupt regularization item 0.2")
parser.add_argument("--droupt_output", default=0.4, type=float, help="droupt regularization item 0.2")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="label smoothing")
parser.add_argument("--CUDA", default=True, type=bool, help="GPU")
parser.add_argument("--model_name", default='CE2T', type=str, help="CE2T ConnectE ETE")
parser.add_argument("--margin", default=1.0, type=float, help="ConnectE ETE’s margine ")
parser.add_argument('--dataset', type=str, default="FB15k", nargs="?",
                    help='Which dataset to use: FB15k, YAGO')
parser.add_argument('--indir', type=str, default='data/FB15k/', help='Input dir of train, test and valid data')
parser.add_argument('--outdir', type=str, default='output/FB15k/', help='Output dir of model')
parser.add_argument('--load', default='True',help='If true, it loads a saved model in outdir and train or evaluate it (default: False)')
args = parser.parse_args()

d = Data(data_dir=args.indir)
