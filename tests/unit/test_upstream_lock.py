import pytest

from bipedlab import upstream


def test_lockfile_pins_full_shas():
    repos = upstream.load_lock()
    assert set(repos) == {"microduck", "microduck_rl", "microduck-lab", "bam"}
    for repo in repos.values():
        assert len(repo.rev) == 40


def test_rejects_branch_names(tmp_path):
    lock = tmp_path / "lock.toml"
    lock.write_text(
        '[repos.x]\nurl = "https://github.com/a/b.git"\nrev = "main"\nlicense = "MIT"\n'
    )
    with pytest.raises(upstream.UpstreamError, match="40-char"):
        upstream.load_lock(lock)


def test_rejects_non_github_urls(tmp_path):
    lock = tmp_path / "lock.toml"
    lock.write_text(
        f'[repos.x]\nurl = "http://evil.example/b.git"\nrev = "{"a" * 40}"\nlicense = "MIT"\n'
    )
    with pytest.raises(upstream.UpstreamError, match="github"):
        upstream.load_lock(lock)
