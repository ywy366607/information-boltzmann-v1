"""Seekable read-only file over HTTP range requests (public bulk connectome tables).

Lets pyarrow read one Arrow IPC record batch at a time from a remote feather
without storing the whole file locally. Blocks are cached in a small LRU.
"""
from collections import OrderedDict
import io
import time

import requests


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, block=8 * 1024 ** 2, cache_blocks=16, session=None):
        self.url, self.block, self.cache_blocks = url, block, cache_blocks
        self.session = session or requests.Session()
        head = self.session.head(url, timeout=30, allow_redirects=True)
        head.raise_for_status()
        self.size = int(head.headers['Content-Length'])
        self.position = 0
        self.cache = OrderedDict()
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def _block(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        start = index * self.block
        end = min(self.size, start + self.block) - 1
        for attempt in range(6):
            try:
                r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=(15, 120))
                if r.status_code != 206 or len(r.content) != end - start + 1:
                    raise IOError(f'bad range response {r.status_code} {len(r.content)}')
                break
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        data = r.content
        self.bytes_fetched += len(data)
        self.cache[index] = data
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        out = bytearray()
        while n > 0:
            index, offset = divmod(self.position, self.block)
            chunk = self._block(index)[offset:offset + n]
            out += chunk
            self.position += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)
