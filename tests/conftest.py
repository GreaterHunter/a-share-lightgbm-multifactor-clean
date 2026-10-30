from pathlib import Path
import pytest

@pytest.fixture
def synthetic_issuer_evidence(tmp_path, monkeypatch):
    """Exercise hashing and dividend arithmetic without redistributing issuer PDFs."""
    from paper_futu import cn_common
    root = tmp_path / 'issuer_fixture'
    (root / 'configs').mkdir(parents=True)
    (root / 'fixture_evidence.txt').write_bytes(b'SYNTHETIC UNIT TEST EVIDENCE')
    source = Path(__file__).parent / 'fixtures/cash_actions_synthetic.json'
    (root / 'configs/verified_cn_cash_actions.json').write_bytes(source.read_bytes())
    monkeypatch.setattr(cn_common, 'ROOT', root)

