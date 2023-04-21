from inspect import currentframe

VISITED_LINES_REGISTRY = []


def you_only_log_once(traceback=0):
    r"""Returns :obj:`True` the first time only, for the given wrapped code, and this for the entire
    life of the current process.
    """
    cf = currentframe()
    caller = cf.f_back
    for _ in range(traceback):
        caller = caller.f_back
    line_no = cf.f_back.f_lineno
    if line_no not in VISITED_LINES_REGISTRY:
        VISITED_LINES_REGISTRY.append(line_no)
        return True
    return False
