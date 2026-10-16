"""Refuse large local jobs when memory pressure risks destabilizing the desktop."""

import ctypes
import os


def available_memory_bytes():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ["total_physical", "available_physical", "total_pagefile", "available_pagefile",
                 "total_virtual", "available_virtual", "available_extended"]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot read available physical memory")
        return status.available_physical
    with open("/proc/meminfo") as source:
        for line in source:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise OSError("Cannot read available memory")


def require_memory(gib: float, purpose: str):
    available = available_memory_bytes() / 1024**3
    if available < gib:
        raise RuntimeError(f"{purpose}需要至少{gib:.1f} GiB可用物理内存，目前仅{available:.2f} GiB；"
                           "先关闭占内存的程序，再重跑。没有强行启动任务。")
    return available
