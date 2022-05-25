"""
Find approximate string matches.
"""

# ------------------------------

import re
import leven
import numpy as np

DELETED_CHARS = ('-')


def normalize(s, deleted_chars=DELETED_CHARS):
    """Return lower-case string with unwanted characters removed."""

    return re.sub('[' + ''.join(deleted_chars) + ']', '', s.lower())


def _distances(query, targets):

    return [leven.levenshtein(query, t) for t in targets]


def match(
        query, targets,
        nbest=1, deleted_chars=DELETED_CHARS, norm=True
):
    """
    Given a query string and a list of target strings, return the best matching
    string, a list of nbest closest matches, sorted by edit distance, and the
    list of indices of the nbest closest matches.
    """

    if norm:

        def _filt(s):
            return normalize(
                unicode(s), deleted_chars=deleted_chars
            )

    else:

        def _filt(s):
            return unicode(s)

    fquery = _filt(query)
    ftargets = np.array(map(_filt, targets))
    distances = _distances(fquery, ftargets)

    argsort = np.argsort(distances)[:nbest]

    return (
        targets[argsort[0]],
        np.array(targets)[argsort].tolist(),
        argsort.tolist(),
        np.array(distances)[argsort].tolist()
    )
