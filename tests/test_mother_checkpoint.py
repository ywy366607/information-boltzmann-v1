import gzip
import io

import torch

from scripts.ib_local.mother_checkpoint import ChunkedWriter


def test_chunked_torch_checkpoint_roundtrip():
    class TrackingStream:
        def __init__(self, stream):
            self.stream = stream
            self.largest = 0

        def write(self, data):
            self.largest = max(self.largest, len(data))
            return self.stream.write(data)

        def flush(self):
            self.stream.flush()

        def tell(self):
            return self.stream.tell()

    raw = io.BytesIO()
    expected = {'table': torch.arange(300_000, dtype=torch.float32), 'cycle': 1894}
    with gzip.GzipFile(fileobj=raw, mode='wb') as compressed:
        tracked = TrackingStream(compressed)
        torch.save(expected, ChunkedWriter(tracked, chunk_size=8192))
        assert tracked.largest <= 8192
    raw.seek(0)
    with gzip.GzipFile(fileobj=raw, mode='rb') as compressed:
        actual = torch.load(compressed, weights_only=False)
    assert actual['cycle'] == expected['cycle']
    assert torch.equal(actual['table'], expected['table'])
