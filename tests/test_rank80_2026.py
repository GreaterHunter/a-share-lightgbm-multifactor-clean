import pytest
from paper_futu.research_rank80_2026 import validate_boundary

def test_first_quarter_uses_pre2026_labels():
    validate_boundary(dict(cutoff='2025-12-31',label_end_max='2025-12-31',train_signal_end='2025-12-16'),
                      dict(cutoff='2025-12-31',start='2026-01-05'))

@pytest.mark.parametrize('key,value',[
    ('label_end_max','2026-01-05'),('train_signal_end','2026-01-05'),('cutoff','2026-01-05')])
def test_cross_boundary_model_rejected(key,value):
    meta=dict(cutoff='2025-12-31',label_end_max='2025-12-31',train_signal_end='2025-12-16')
    meta[key]=value
    with pytest.raises(AssertionError):
        validate_boundary(meta,dict(cutoff='2025-12-31',start='2026-01-05'))

def test_model_not_available_before_execution():
    with pytest.raises(AssertionError):
        validate_boundary(dict(cutoff='2026-03-31',label_end_max='2026-03-31',train_signal_end='2026-03-13'),
                          dict(cutoff='2026-03-31',start='2026-03-31'))
