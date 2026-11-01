"""Record the local full-pixel reproduction's elapsed time and peak memory."""
from pathlib import Path
import json
import platform
import resource
import subprocess
import time

from reproduce_panel import main

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    start=time.perf_counter()
    main()
    elapsed=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system()!='Darwin':
        peak*=1024
    machine=platform.machine()
    if platform.system()=='Darwin':
        try:
            machine=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],
                                            text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            pass
    report={'task':'Recompute all 3861 county-week pixel counts and 39 state aggregates from locally stored packed masks; source-file download and preparation excluded.',
            'hardware':machine,'operating_system':platform.system(),'python':platform.python_version(),
            'elapsed_seconds':elapsed,'peak_resident_bytes':peak,
            'cache_note':'Public inputs already stored locally. No cold OS-cache claim or relative algorithm speed comparison.'}
    (ROOT/'results/runtime_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
