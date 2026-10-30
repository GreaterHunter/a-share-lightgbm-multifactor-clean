"""Memory-gated execution only; does not modify the frozen scientific module."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from filelock import FileLock

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/experiments/s6_barra_overlay_asof_20261013'
MODULE='paper_futu.research_s6_barra_overlay'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def save(p,x):
    t=Path(p).with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=True,indent=2),encoding='utf-8');t.replace(p)

def main():
    binding=dict(script_sha256=sha(__file__),scientific_protocol_sha256=sha(OUT/'protocol.json'),
        max_workers=2,start_free_gib=1.5,worker_floor_gib=.5,
        overrides_original_runtime_workers_only=True,
        scope='Scheduling only, existing 45 mutually exclusive accounts; scientific sources unchanged')
    path=OUT/'execution_scheduler.json'
    if path.exists() and read(path)!=binding:raise ValueError('Scheduler binding changed')
    save(path,binding)
    with FileLock(str(OUT/'controller.lock'),timeout=0):
        conflicts=[]
        for p in psutil.process_iter(['pid','cmdline']):
            cmd=p.info['cmdline'] or []
            if p.pid!=os.getpid() and MODULE in cmd:conflicts.append((p.pid,cmd))
        if conflicts:raise RuntimeError('Existing research processes: '+str(conflicts))
        tasks=[(c,o,b) for b in (0,10,20) for o in range(5) for c in ('R0','R1','R2')]
        def marker(task):
            c,o,b=task;return OUT/f'accounts/history_{c}_f{b}_{o}/complete.json'
        queue=[t for t in tasks if not marker(t).exists()];completed=len(tasks)-len(queue)
        running={};wait_since=None;error=None
        while queue or running:
            for pid,(p,t,log) in list(running.items()):
                code=p.poll()
                if code is None:continue
                log.close();del running[pid]
                if code or not marker(t).exists():error=dict(task=t,exit_code=code);queue=[]
                else:
                    completed+=1;print('Completed',completed,'/45',t,flush=True)
            if error and not running:raise RuntimeError(str(error))
            free=psutil.virtual_memory().available/2**30
            if queue and not error and len(running)<2 and free>=1.5:
                t=queue.pop(0);c,o,b=t
                log=(OUT/f'logs/{c}_f{b}_{o}.log').open('a',encoding='utf-8')
                p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m',MODULE,'--stage','account',
                    '--case',c,'--offset',str(o),'--cost',str(b)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'},
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                running[p.pid]=(p,t,log);wait_since=None
                # Let the new interpreter establish its memory footprint before
                # deciding whether a second process fits.
                time.sleep(12)
            elif not running and queue:
                wait_since=wait_since or time.time()
                if time.time()-wait_since>1800:raise RuntimeError('Memory below startup floor for 30 minutes; no automatic restart')
            active=[]
            for pid,(p,t,log) in running.items():
                try:active.append(dict(pid=pid,task=t,create_time=psutil.Process(pid).create_time()))
                except psutil.NoSuchProcess:pass
            save(OUT/'status.json',dict(state='running' if running else 'waiting_memory',completed=completed,total=45,
                pid=os.getpid(),active=active,available_gib=free,updated=time.time()))
            time.sleep(2)
        r=subprocess.run([sys.executable,'-X','utf8','-u','-m',MODULE,'--stage','report'],cwd=ROOT)
        if r.returncode:raise RuntimeError('Report failed')
        complete=read(OUT/'completion.json')
        complete['execution_scheduler_sha256']=sha(OUT/'execution_scheduler.json')
        complete['scheduler_script_sha256']=sha(__file__)
        save(OUT/'completion.json',complete)
        save(OUT/'status.json',dict(state='complete',completed=45,total=45,updated=time.time()))

if __name__=='__main__':main()
