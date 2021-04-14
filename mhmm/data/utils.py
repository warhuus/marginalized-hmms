def make_lengths(opt: dict) -> list:
    """ Make list of lengths of sequences """
    if opt['data'] == 'hcp':
        return opt.get('subjects') * [405]
    else:
        N_seq = opt.get('N_seq')
        return [opt['N']] * N_seq if N_seq is not None else [opt['N']]