import pandas as pd
import pytest
from paper_futu.fetch_barra_metadata import industry_rows,share_events


def test_share_change_waits_for_both_effective_date_and_notice_next_day():
    rows=[dict(SECURITY_CODE='600000',END_DATE='2025-01-01',NOTICE_DATE='2025-01-10',
        TOTAL_SHARES=120.,LISTED_A_SHARES=80.,LIMITED_A_SHARES=20.)]
    e=share_events(rows,'SH600000')
    assert e.usable_date.iloc[0]==pd.Timestamp('2025-01-11')
    assert e.a_shares.iloc[0]==100. # Non-A shares are not valued at the A-share price.


def test_ambiguous_share_count_is_not_fabricated_and_future_notice_is_excluded():
    rows=[dict(SECURITY_CODE='600000',END_DATE='2025-01-01',NOTICE_DATE='2025-01-10',
        TOTAL_SHARES=120.,LISTED_A_SHARES=80.,LIMITED_A_SHARES=None),
        dict(SECURITY_CODE='600000',END_DATE='2026-10-01',NOTICE_DATE='2026-10-20',
        TOTAL_SHARES=150.,LISTED_A_SHARES=150.,LIMITED_A_SHARES=0.)]
    assert share_events(rows,'SH600000').empty


def test_industry_new_taxonomy_is_not_backfilled_to_ipo():
    p={'records':[dict(F001V='008003',F004V='Bank',VARYDATE='1999-01-01')]}
    e=industry_rows(p,'SH600000',93)
    assert e.effective_date.iloc[0]==pd.Timestamp('2021-11-15')
    assert e.available_date.iloc[0]==pd.Timestamp('2022-02-16')
    assert not e.first_publication_verified.iloc[0]


def test_only_selected_industry_standards_are_combined():
    p={'records':[dict(F001V='008002',F004V='Current',VARYDATE='2010-01-01'),
        dict(F001V='008018',F004V='Old',VARYDATE='2020-01-01'),
        dict(F001V='008003',F004V='New',VARYDATE='2022-01-01')]}
    e=industry_rows(p,'SH600000',1)
    assert e.industry.tolist()==['SW:Old','SW:New']
    assert e.available_date.tolist()==[pd.Timestamp('2020-01-02'),pd.Timestamp('2022-01-02')]
