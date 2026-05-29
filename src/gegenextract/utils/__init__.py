import time
from contextlib import contextmanager


@contextmanager
def timer(name: str = "operation"):
    t0 = time.time()
    try:
        yield
    finally:
        print(f"{name} took {time.time() - t0:.3f}s")
