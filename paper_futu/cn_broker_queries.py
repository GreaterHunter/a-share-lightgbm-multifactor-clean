"""Bounded position-query budget plus a short-lived whole-account snapshot."""
from collections import deque
import time


class WindowBudget:
    def __init__(self, calls=8, seconds=30., clock=time.monotonic, sleep=time.sleep):
        self.calls, self.seconds, self.clock, self.sleep = calls, seconds, clock, sleep
        self.history = deque()

    def acquire(self):
        while True:
            now = self.clock()
            while self.history and now-self.history[0] >= self.seconds:
                self.history.popleft()
            if len(self.history) < self.calls:
                self.history.append(now)
                return
            self.sleep(max(.01, self.seconds-(now-self.history[0])+.05))


POSITION_BUDGET = WindowBudget()
FUNDS_BUDGET = WindowBudget()
ORDERS_BUDGET = WindowBudget()
SUBMIT_BUDGET = WindowBudget(calls=12)
CANCEL_BUDGET = WindowBudget(calls=16)


def bounded_read(call, budget, operation, refresh=True):
    """One retry only for explicit throttling; never wrap an order submission."""
    from paper_futu.smoke_order import require_ok
    if refresh:
        budget.acquire()
    result = call()
    if result[0] != 0 and ("频率" in str(result[1]) or "too frequent" in str(result[1]).lower()):
        time.sleep(30.1)
        if refresh:
            budget.acquire()
        result = call()
    return require_ok(result, operation)


def query_positions(trade, account_id, **kwargs):
    from futu import TrdEnv
    from paper_futu.smoke_order import require_ok
    POSITION_BUDGET.acquire()
    result = trade.position_list_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id, refresh_cache=True, **kwargs)
    # Other OpenD clients share the server quota. Only explicit throttling gets
    # one bounded wait/retry; order submissions are NEVER retried here.
    if result[0] != 0 and ("频率" in str(result[1]) or "too frequent" in str(result[1]).lower()):
        time.sleep(30.1)
        POSITION_BUDGET.acquire()
        result = trade.position_list_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id, refresh_cache=True, **kwargs)
    return require_ok(result, "查询模拟持仓")


class PositionCache:
    def __init__(self, fetch, initial=None, ttl=10., clock=time.monotonic):
        self.fetch, self.frame, self.ttl, self.clock = fetch, initial, ttl, clock
        self.stamp = clock() if initial is not None else -float("inf")

    def get(self, force=False):
        if force or self.frame is None or self.clock()-self.stamp >= self.ttl:
            self.frame = self.fetch()
            self.stamp = self.clock()
        return self.frame
