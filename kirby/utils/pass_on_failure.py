import logging

from src.utils import you_only_log_once

log = logging.getLogger(__name__)


def pass_on_failure(f):
    r"""Doesn't raise error if function fails, returns None and logs Error once.
    This is useful when a non-necessary function fails and would interrupt the process.
    """
    def applicator(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            if you_only_log_once(traceback=1):
                log.warning('Call to function %r failed: %r.', f.__name__, e)
    return applicator
