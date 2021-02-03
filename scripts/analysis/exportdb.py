import os
import sys
import argparse
from io import BytesIO
from itertools import chain

import torch

import json
from bson.json_util import dumps as bson_dumps
from incense import ExperimentLoader


parser = argparse.ArgumentParser("Export experiment results to db for anlysis")

parser.add_argument('--db', type=str)
parser.add_argument('--expids', nargs='+', type=int)
parser.add_argument('-r', action='store_true')
parser.add_argument('--path', type=str, default='dump')
parser.add_argument('--mongo_uri', type=str, default='127.0.0.1')


def export(exp_id, loader, path):
    path = path.rstrip('/') + '/{}/'.format(exp_id)
    os.makedirs(path, exist_ok=True)

    exp = loader.find_by_id(exp_id)
    metrics = exp.metrics

    exp_dict = exp.to_dict()

    cout = exp_dict.pop('captured_out')
    config = exp_dict.pop('config')

    with open(path + 'cout.txt', mode='w') as f:
        f.write(cout)
        
    with open(path + 'run.json', mode='w') as f:
        json.dump(json.loads(bson_dumps(exp_dict)), f, indent=4)
        
    with open(path + 'config.json', mode='w') as f:
        json.dump(json.loads(bson_dumps(config)), f, indent=4)
        
    with open(path + 'metrics.json', mode='w') as f:
        json.dump(json.loads(bson_dumps(metrics)), f, indent=4)
        
    for k in exp.artifacts:
        artifact_content = torch.load(BytesIO(exp.artifacts[k].content))

        with open(path + k, mode='wb') as f:
            torch.save(artifact_content, f)


if __name__ == "__main__":
    args = parser.parse_args()

    path = args.path

    loader = ExperimentLoader(
        mongo_uri=args.mongo_uri, 
        db_name=args.db
    )

    if args.r:
        exp_ids_starts = args.expids[::2]
        exp_ids_ends = args.expids[1::2]
        expids = chain(*[range(s, e + 1) for s, e in 
                            zip(exp_ids_starts, exp_ids_ends)])

    else:
        expids = args.expids

    for exp_id in expids:
        export(exp_id, loader, path)
