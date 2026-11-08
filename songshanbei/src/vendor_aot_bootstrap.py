"""Run the vendor Triton backend with older LLVM-compatible GEP syntax."""

import functools
import re

from triton.backends.amd.compiler import HIPBackend


def install():
    original = HIPBackend.make_amdgcn

    @functools.wraps(original)
    def compatible(ir, metadata, options):
        def preserve_inbounds(match):
            return "getelementptr inbounds " if "inbounds" in match[0] else "getelementptr "

        normalized = re.sub(
            r"\bgetelementptr\s+(?:(?:inbounds|nuw|nusw)\s+)+",
            preserve_inbounds,
            ir,
        )
        return original(normalized, metadata, options)

    HIPBackend.make_amdgcn = staticmethod(compatible)


install()
