#!/usr/bin/env python
"""Print PyTorch CUDA status for pytorch"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

import torch


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: A list of argument strings to use instead of sys.argv.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run script.

    Args:
        argv: A list of argument strings to use instead of sys.argv.
    """
    args = parse_args(argv)
    del args
    cuda_is_available = torch.cuda.is_available()
    print("CUDA is available:", cuda_is_available)
    if not cuda_is_available:
        return
    for i in range(torch.cuda.device_count()):
        print(i, ": ", torch.cuda.get_device_name(i), sep='')


if __name__ == "__main__":
    main()
