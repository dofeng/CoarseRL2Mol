import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from trajectory_pipeline import _decode_atom_ids, _encode_atom_ids, discover_frames


def test_atom_id_storage_round_trip():
    atom_ids = [1, 2, 10, 999999]
    assert _decode_atom_ids(_encode_atom_ids(atom_ids), len(atom_ids)) == atom_ids


def test_frame_discovery_defaults_to_filename_frame_number():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        later = root / "bond1.reaxff"
        earlier = root / "bond999.reaxff"
        later.write_text("# Timestep 200\n", encoding="utf-8")
        earlier.write_text("# Timestep 100\n", encoding="utf-8")
        frames = discover_frames([str(root)], "*.reaxff")
        assert [Path(item["source_path"]).name for item in frames] == ["bond1.reaxff", "bond999.reaxff"]
        assert [item["timestep"] for item in frames] == [200, 100]
        assert [item["frame_id"] for item in frames] == [0, 1]


def test_frame_discovery_allows_duplicate_header_timesteps():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "bond2000.reaxff").write_text("# Timestep 200000\n", encoding="utf-8")
        (root / "bond2001.reaxff").write_text("# Timestep 200000\n", encoding="utf-8")
        frames = discover_frames([str(root)], "*.reaxff")
        assert [Path(item["source_path"]).name for item in frames] == [
            "bond2000.reaxff", "bond2001.reaxff",
        ]
        assert [item["timestep"] for item in frames] == [200000, 200000]


def test_frame_discovery_can_explicitly_use_header_timestep():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "bond1.reaxff").write_text("# Timestep 200\n", encoding="utf-8")
        (root / "bond999.reaxff").write_text("# Timestep 100\n", encoding="utf-8")
        frames = discover_frames([str(root)], "*.reaxff", "timestep")
        assert [item["timestep"] for item in frames] == [100, 200]
