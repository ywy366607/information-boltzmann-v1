"""Bounded, lossy visualization IPC. Browser and disk work never run in training."""
import multiprocessing as mp
from pathlib import Path
import queue
import time


def _worker(messages, directory, port, html):
    from .telemetry import TelemetryHub
    from torch.utils.tensorboard import SummaryWriter
    hub = TelemetryHub(history_len=2000, live_file=Path(directory)/"live_state.json")
    writer = SummaryWriter(str(Path(directory)/"tensorboard"))
    if port:
        hub.start_server(port, Path(html))
    try:
        while True:
            frame = messages.get()
            if frame is None:
                break
            step = frame["event"]
            for key in ("ce", "mean_nll", "lr"):
                writer.add_scalar("learning/"+key, frame[key], step)
            for key,value in frame["metrics"].items():
                writer.add_scalar("phase/"+key, value, step)
            hub.push_frame(frame)
    finally:
        writer.close()
        hub.stop_server()


class AsyncMonitor:
    def __init__(self, directory, port=0, html="present/information_boltzmann_monitor.html",
                 max_hz=2., capacity=4):
        context = mp.get_context("spawn")
        self.messages = context.Queue(maxsize=capacity)
        self.process = context.Process(target=_worker,args=(self.messages,str(directory),port,html),daemon=True)
        self.process.start()
        self.interval = 1/max_hz
        self.last = float("-inf")
        self.dropped = 0

    def due(self):
        return time.monotonic()-self.last >= self.interval

    def push_frame(self, frame):
        self.last=time.monotonic()
        try:
            self.messages.put_nowait(frame)
        except queue.Full:
            self.dropped+=1

    def close(self):
        # Blocking is allowed at shutdown, never inside a model step.
        try:
            self.messages.put(None,timeout=10)
        except queue.Full:
            self.process.terminate()
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.messages.close()
