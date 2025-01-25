from pde_base import PDEBase
from CE_KH_openfoam.CE_KH import CE_KH
from CE_CRP_openfoam.CE_CRP import CE_CRP

def get_pde_variant(pde_name: str) -> PDEBase:
    """
    Return an instance of the correct PDE class based on pde_name.
    Add more 'elif' blocks if you have more PDE variants.
    """
    if pde_name.lower() == "ce_kh":
        return CE_KH(), "CE_KH_openfoam"
    elif pde_name.lower() == "ce_crp":
        return CE_CRP(), "CE_CRP_openfoam"
    else:
        raise ValueError(f"Unknown PDE variant: {pde_name}")