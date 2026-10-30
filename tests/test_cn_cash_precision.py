import pytest
from paper_futu.independent_cn_market import verified_cash_deduction


@pytest.mark.parametrize('cash',['1.0992','1.099231'])
def test_luxshare_issuer_cash_and_verified_provider_precision(cash):
    assert verified_cash_deduction('SZ002475','2026-10-22',{'fh_sh':cash})==pytest.approx(.1096588)


@pytest.mark.parametrize('cash',['1.0993','1.1','1.096588'])
def test_other_cash_values_still_fail(cash):
    with pytest.raises(ValueError,match='identity/evidence'):
        verified_cash_deduction('SZ002475','2026-10-22',{'fh_sh':cash})


def test_no_unverified_or_wrong_date_exception():
    assert verified_cash_deduction('SZ002475','2026-10-21',{'fh_sh':'1.0992'}) is None
    assert verified_cash_deduction('SZ002476','2026-10-22',{'fh_sh':'1.0992'}) is None

pytestmark = pytest.mark.usefixtures("synthetic_issuer_evidence")
