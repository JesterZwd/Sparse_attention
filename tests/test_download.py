import hashlib
import io
import urllib.error

import pytest

from scripts.download_model import download_file


class Response(io.BytesIO):
    status = 200


@pytest.mark.parametrize("partial", [b"checkpoint", b"corrupted!", b"oversized checkpoint"])
def test_complete_partial_is_promoted_or_restarted(tmp_path, monkeypatch, partial):
    content = b"checkpoint"
    destination = tmp_path / "weights"
    destination.with_name("weights.part").write_bytes(partial)
    class Opener:
        def open(self, request, timeout):
            return Response(content)
    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    download_file({"Size": len(content), "Sha256": hashlib.sha256(content).hexdigest()},
                  destination, "https://example.invalid/checkpoint")
    assert destination.read_bytes() == content
    assert not destination.with_name("weights.part").exists()


def test_http_416_restarts_partial_download(tmp_path, monkeypatch):
    content = b"checkpoint"
    destination = tmp_path / "weights"
    destination.with_name("weights.part").write_bytes(b"bad")
    class Opener:
        def open(self, request, timeout):
            if request.has_header("Range"):
                raise urllib.error.HTTPError(request.full_url, 416, "range rejected", {}, None)
            return Response(content)
    monkeypatch.setattr("urllib.request.build_opener", lambda *_: Opener())
    monkeypatch.setattr("time.sleep", lambda *_: None)
    download_file({"Size": len(content), "Sha256": hashlib.sha256(content).hexdigest()},
                  destination, "https://example.invalid/checkpoint")
    assert destination.read_bytes() == content
