import pytest

from bipedlab import policies


def test_lock_is_pinned():
    lock = policies.load_lock()
    assert lock.repo == "pollen-robotics/microduck-policies"
    assert len(lock.revision) == 40
    assert "alpha_walking.onnx" in lock.files
    assert all(len(h) == 64 for h in lock.files.values())
    assert lock.url("alpha_walking.onnx").startswith(
        "https://huggingface.co/pollen-robotics/microduck-policies/resolve/088524a"
    )


def test_checksum_mismatch_is_rejected(tmp_path):
    lock = policies.PolicyLock(repo="x/y", revision="a" * 40, files={"p.onnx": "0" * 64})
    (tmp_path / "p.onnx").write_bytes(b"tampered")
    with pytest.raises(policies.PolicyFileError, match="sha256"):
        policies.verified_path("p.onnx", lock, cache=tmp_path)


def test_missing_file_hint(tmp_path):
    lock = policies.PolicyLock(repo="x/y", revision="a" * 40, files={"p.onnx": "0" * 64})
    with pytest.raises(policies.PolicyFileError, match="fetch_policies"):
        policies.verified_path("p.onnx", lock, cache=tmp_path)
