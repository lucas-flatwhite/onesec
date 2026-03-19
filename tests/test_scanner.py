from pathlib import Path
import pytest
from onesec.scanner import scan


def test_scan_finds_video_files(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mov").touch()
    (tmp_path / "c.txt").touch()

    paths = scan(tmp_path)
    names = {p.name for p in paths}
    assert "a.mp4" in names
    assert "b.mov" in names
    assert "c.txt" not in names


def test_scan_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.mp4").touch()

    paths = scan(tmp_path, recursive=True)
    assert any(p.name == "c.mp4" for p in paths)


def test_scan_non_recursive_default(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.mp4").touch()
    (tmp_path / "a.mp4").touch()

    paths = scan(tmp_path)
    names = {p.name for p in paths}
    assert "a.mp4" in names
    assert "c.mp4" not in names


def test_scan_empty_folder(tmp_path):
    assert scan(tmp_path) == []


def test_scan_single_file(tmp_path):
    f = tmp_path / "v.mp4"
    f.touch()
    paths = scan(f)
    assert len(paths) == 1
    assert paths[0] == f
