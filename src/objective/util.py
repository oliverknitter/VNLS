def get_hamiltonian(pb_type, **kwargs):
    if pb_type in ['maxcut']:
        from .maxcut import MaxCut
        return MaxCut(**kwargs)
    elif pb_type in ['vqls']:
        from .vqls import VQLS
        return VQLS(**kwargs)
    else:
        raise "Problem type unspecified!"
