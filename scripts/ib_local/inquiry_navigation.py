"""Private corpus responder for explicit read/ask comparisons."""
import re

POLICIES = ('suffix_read_control', 'inquire_terms_v1', 'inquire_read_control')
HEADER = 64


def address(query, corpus, window, cursor):
    # The whole query is eligible. Prefer the longest complete lexical term;
    # ties follow query order, then nearest continuation as in the old world.
    terms = dict.fromkeys(re.findall(rb'[a-z0-9]{2,}', query.lower()))
    for term in sorted(terms, key=len, reverse=True):
        best, count = -1, 0
        for hit in re.finditer(rb'(?<![a-z0-9])'+term+rb'(?![a-z0-9])', corpus):
            pos = hit.end()
            if pos + window > len(corpus): break
            count += 1
            if best < 0 or abs(pos-cursor) < abs(best-cursor): best = pos
        if best >= 0: return best, len(term), count, abs(best-cursor), term
    return -1, 0, 0, -1, b''


def header(asked, term):
    result = b'READ' if not asked else b'HIT '+term[:16] if term else b'NO HIT; READ'
    return (b'[silent=read; speak=ask corpus] '+result).ljust(HEADER-1)+b'\n'
