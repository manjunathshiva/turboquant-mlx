"""SafetensorsExpertReader.close() must stay silent at interpreter teardown.

close() runs from __del__. At teardown module globals such as ``os`` may already
be None, so ``os.close`` raises AttributeError -- which the handler used to miss
(it caught only OSError), printing an ignored-exception traceback on every exit
of every streaming script.
"""

import os as real_os

import turboquant_mlx.stream.safetensors_reader as sr


def test_close_survives_os_global_being_none(monkeypatch, tmp_path):
    p = tmp_path / "blob"
    p.write_bytes(b"x")
    fd = real_os.open(p, real_os.O_RDONLY)
    reader = object.__new__(sr.SafetensorsExpertReader)
    reader._fds = [fd]
    monkeypatch.setattr(sr, "os", None)
    try:
        reader.close()  # must not raise
    finally:
        monkeypatch.undo()
        real_os.close(fd)
    assert reader._fds == []
