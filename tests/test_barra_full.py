import numpy as np
import pandas as pd
import pytest
from paper_futu.research_barra_full import hac_mean,fixed_nonlinear_size
from paper_futu.research_barra import save,sha


def test_hac_mean_constant_daily_residual_has_correct_annual_scale():
    r=hac_mean(np.repeat(.0001,500))
    assert r['annual_arithmetic_mean']==pytest.approx(.0252)
    assert r['ci_low']==pytest.approx(.0252)
    assert r['ci_high']==pytest.approx(.0252)


def test_hac_rejects_nonfinite_or_too_short_input():
    with pytest.raises(ValueError):hac_mean([.1]*5)
    with pytest.raises(ValueError):hac_mean([np.nan]*500)


@pytest.mark.parametrize('equal',[True,False])
def test_nonlinear_size_transform_is_orthogonal_and_resumable(tmp_path,equal):
    n=100;idx=pd.MultiIndex.from_arrays([[pd.Timestamp('2025-01-01')]*n,[f's{i}' for i in range(n)]],names=['datetime','instrument'])
    x=np.linspace(-2,2,n);f=pd.DataFrame({'size':x,'nonlinear_size':x**3,'capital':np.exp(x)*1e9},index=idx)
    path=tmp_path/'exposures.parquet';f.to_parquet(path)
    save(tmp_path/'data_complete.json',{'status':'complete','artifacts':{str(path):sha(path)}})
    fixed_nonlinear_size(tmp_path,equal);result=pd.read_parquet(path)
    w=np.ones(n) if equal else f.capital.reindex(result.index).to_numpy()
    assert abs(np.average(result['size']*result.nonlinear_size,weights=w))<1e-9
    expected=sha(path);fixed_nonlinear_size(tmp_path,equal);assert sha(path)==expected
