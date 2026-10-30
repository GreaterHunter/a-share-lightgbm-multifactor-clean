import pytest
from paper_futu.research_rank_buffer_futu_fees import fee,buy_quantity,liquidation_breakeven,engine

@pytest.mark.parametrize('value,buy,expected',[(100000,True,55.87),(100000,False,155.87),(50000,True,35.435),(50000,False,85.435),(100,True,18.04),(0,True,0)])
def test_fee(value,buy,expected):assert fee(value,buy)==pytest.approx(expected)

@pytest.mark.parametrize('code,factor,money',[('SZ300308',1,10000),('SH688001',.2,50000),('SZ000001',.01,500)])
def test_affordability(code,factor,money):
    price=10*factor;q=buy_quantity(code,money,price,factor)
    assert q*price+fee(q*price,True)<=money
    step=1 if code.startswith('SH688') else 100
    nextq=q+step/factor
    assert nextq*price+fee(nextq*price,True)>money

@pytest.mark.parametrize('basis',[0,100,1000,100000])
def test_breakeven(basis):
    p=liquidation_breakeven(100,basis,0)
    assert 100*p-fee(100*p,False)==pytest.approx(basis,abs=1e-8)

def test_isolation():
    from paper_futu import rank_buffer_10_20_engine as old
    assert old.fee(100000,True)==50
    assert engine(True).fee(100000,True)==pytest.approx(55.87)
    assert old.fee(100000,True)==50
    assert engine(False).fee(100000,False)==pytest.approx(155.87)
