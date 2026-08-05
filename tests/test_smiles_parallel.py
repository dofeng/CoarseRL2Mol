from pathlib import Path

import pytest

from model.data.smiles import process_input_files


@pytest.fixture
def chembl_input(tmp_path: Path) -> Path:
    source = tmp_path / "chemreps.tsv"
    source.write_text(
        "chembl_id\tcanonical_smiles\tstandard_inchi\n"
        "CHEMBL1\tCCO\t-\n"
        "CHEMBL2\tCCCl\t-\n"
        "CHEMBL3\tCCCCCCCC\t-\n"
        "CHEMBL4\tnot-a-smiles\t-\n",
        encoding="utf-8",
    )
    return source


def _run(source: Path, outdir: Path, num_workers: int) -> int:
    return process_input_files(
        inputs=[str(source)],
        outdir=str(outdir),
        min_heavy=1,
        max_heavy=20,
        overwrite=False,
        num_workers=num_workers,
        batch_size=2,
    )


def test_parallel_output_matches_single_process(
    chembl_input: Path,
    tmp_path: Path,
) -> None:
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"

    assert _run(chembl_input, serial_dir, num_workers=1) == 2
    assert _run(chembl_input, parallel_dir, num_workers=2) == 2

    serial = {path.name: path.read_text() for path in serial_dir.iterdir()}
    parallel = {path.name: path.read_text() for path in parallel_dir.iterdir()}
    assert parallel == serial == {
        "CHEMBL1.txt": "CCO\n",
        "CHEMBL3.txt": "CCCCCCCC\n",
    }


def test_parallel_run_skips_existing_outputs(
    chembl_input: Path,
    tmp_path: Path,
) -> None:
    outdir = tmp_path / "parallel"

    assert _run(chembl_input, outdir, num_workers=2) == 2
    assert _run(chembl_input, outdir, num_workers=2) == 0


@pytest.mark.parametrize(
    ("num_workers", "batch_size"),
    [(0, 2), (1, 0)],
)
def test_invalid_parallel_settings_are_rejected(
    chembl_input: Path,
    tmp_path: Path,
    num_workers: int,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        process_input_files(
            inputs=[str(chembl_input)],
            outdir=str(tmp_path / "out"),
            min_heavy=1,
            max_heavy=20,
            overwrite=False,
            num_workers=num_workers,
            batch_size=batch_size,
        )
