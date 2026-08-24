import json
import argparse
import logging
from pathlib import Path
import recsiam.data as data


def main(cmdline):
    if not cmdline.ambiguity:
        dd = data.descriptor_from_filesystem(cmdline.dataset)
    else:
        dd = data.ambiguity_descriptor_from_filesystem(cmdline.dataset)

    with Path(cmdline.json).open("w") as of:
        json.dump(dd, of, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="root folder for the target dataset")
    parser.add_argument("json", type=str,
                        help="path to store json")
    parser.add_argument("--ambiguity", action='store_true',
                        help="load as ambiguity dataset")

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
