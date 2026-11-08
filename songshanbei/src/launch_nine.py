"""Load the Hygon NineToothed dispatch bridge before the official FL2VA entry."""

import ctypes
import os
import runpy
import sys
from pathlib import Path

import infinicore


def register():
    library = ctypes.CDLL(os.environ["H3_NINE_MHA_BRIDGE"])
    configure = library.h3_register_nine_mha
    configure.argtypes = []
    configure.restype = None
    configure()
    print(f"H3_NINE_MHA_REGISTERED core={infinicore.__file__}", flush=True)
    return library


if __name__ == "__main__":
    entry = Path(sys.argv[1]).resolve(strict=True)
    bridge = register()
    sys.argv = [str(entry), *sys.argv[2:]]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name="__main__")
