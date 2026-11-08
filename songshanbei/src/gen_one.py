import ninetoothed
import infiniop.ninetoothed.build as nb
from infiniop.ops.flash_attention.ninetoothed import flash_attention as fa

grid = {
    "with_kv_cache":  (0,),
    "emb_dim":        (128,),
    "is_causal":      (0,),
    "with_attn_mask": (0,),
    "causal_variant": (fa.CausalVariant.LOWER_RIGHT,),
    "dtype":          (ninetoothed.bfloat16,),
    "block_size_m":   (256,),
    "block_size_n":   (64,),
}
nb.build(fa.premake, grid, caller="cuda", op_name="flash_attention",
         output_dir=nb.BUILD_DIRECTORY_PATH)
print("ONE DONE")
