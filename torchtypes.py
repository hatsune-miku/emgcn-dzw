from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

# Should be called globally upon each import
patch_typeguard()
