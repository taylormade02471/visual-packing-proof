from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_reports_measured_evidence_and_limits() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "29.896 ms" in readme
    assert "97.14%" in readme
    assert ">99.9%" in readme
    assert "production target" in readme.lower()
    assert "/api/packing/verify" in readme


def test_handover_documents_exist() -> None:
    required = (
        "docs/ARCHITECTURE.md",
        "docs/DATA_DICTIONARY.md",
        "docs/RETRAINING_PLAYBOOK.md",
        "docs/REWORK_PROPOSAL.md",
    )
    for relative_path in required:
        document = ROOT / relative_path
        assert document.is_file(), relative_path
        assert len(document.read_text(encoding="utf-8")) > 300, relative_path


def test_container_contract_is_cloud_run_compatible() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uvicorn" in dockerfile
    assert "${PORT:-8080}" in dockerfile
    assert "MobileNetV3Encoder" in dockerfile
    assert (ROOT / ".dockerignore").is_file()
