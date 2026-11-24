"""Atomic publication with bounded retries for transient Windows file locks."""
import time
from pathlib import Path


def publish_checkpoint(temporary, target):
    temporary, target = Path(temporary), Path(target)
    if temporary.resolve().parent != target.resolve().parent:
        raise ValueError('atomic checkpoint must remain in its output directory')
    for attempt in range(6):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise  # Preserve the complete temporary and previous commit.
            time.sleep(.05 * (2**attempt))
