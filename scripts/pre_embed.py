import argparse
import logging
import json
from pathlib import Path

import torch
import numpy as np
import lz4.frame

import recsiam.embeddings  as emb
import recsiam.models as models
import recsiam.data as data
import recsiam.utils  as utils

def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset_model(cmdline, target_dataset):
    dataset = data.dataset_from_filesystem(target_dataset)
    if cmdline.mask_dataset is not None:
        mask_dataset = data.VideoDataSet(data.mask_from_filesystem(cmdline.mask_dataset, cmdline.mask_name))

        dataset = data.MaskedDataset(dataset, mask_dataset) 

    module_list = [utils.default_image_normalizer(),
                       cmdline.cnn_embedding(pretrained=True),
                       models.BatchFlattener()]
    model = torch.nn.Sequential(*module_list)

    if cmdline.use_gpu:
        model.cuda()

    return dataset, model



def main(cmdline):

    if cmdline.seed is not None:
        set_torch_seeds(cmdline.seed)

    dataset, model = get_dataset_model(cmdline, cmdline.dataset)
    model.eval()

    outf = Path(cmdline.outfolder)
    outf.mkdir(exist_ok=True)
    if len(dataset.info) > 0:
        (outf / "metadata.json").write_text(json.dumps(dataset.info))

    for data, seq in dataset.gen_embed_dataset():
        seq  = Path(seq[0]).parent
        obj_out = outf / seq.parent.name / seq.name

        embedded = simple_forward(data, cmdline.batch_size, model)

        obj_out.mkdir(parents=True, exist_ok=True)
        if not cmdline.compressed:
            np.save(obj_out / "data.npy", embedded)
        else:
            with lz4.frame.open(str(obj_out / "data.npy.lz4"), mode="wb", compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC) as f:
                np.save(f, embedded)


def simple_forward(input_data, batch_size, model):

    batches = np.array_split(input_data, int(np.ceil(input_data.shape[0] / batch_size)))
    on_cuda = next(model.parameters()).is_cuda
    result = [] 

    with torch.no_grad():
        for batch in batches:
            tns = torch.from_numpy(batch)
            if on_cuda:
                tns = tns.cuda()

            result.append(model.forward(tns).detach().cpu().numpy())

    return np.concatenate(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="directory containing the dataset to use")
    parser.add_argument("outfolder", type=str,
                        help="ouput directory for the pre embeded dataset")
    parser.add_argument("--mask-dataset", type=str, default=None,
                        help="ouput directory masking dataset")
    parser.add_argument("--mask-name", type=str, default="unary.png",
                        help="name of the mask to appy")
    parser.add_argument("-c", "--use-gpu", action='store_true',
                        help="toggles gpu")
    parser.add_argument("-b", "--batch-size", type=int, default=10,
                        help="batch size")
    parser.add_argument("--cnn-embedding", type=emb.get_embedding,
                        default=emb.squeezenet1_1embedding,
                        help="Embedding network to use")
    parser.add_argument("-z", "--compressed", action='store_true',
                        help="store a compressed version of the embedding")
    parser.add_argument("-s", "--seed", type=int, default=None,
                        help="Seed to use, default random init")
#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    main(args)
