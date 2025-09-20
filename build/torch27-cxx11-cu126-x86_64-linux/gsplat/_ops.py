import torch
from . import _gsplat_20250920003636
ops = torch.ops._gsplat_20250920003636

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_gsplat_20250920003636::{op_name}"