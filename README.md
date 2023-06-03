# TryTune

Heterogeneous System Model Pipeline Scheduling Framework with Triton Inference Server as Backend

# Install & Run
```bash
$ git clone https://github.com/cmpark0126/trytune-py
$ cd trytune-py
$ pip install -r requirements.txt
$ pip install . # install trytune
$ uvicorn fastapi.main:app --host 0.0.0.0 --port 80 --log-level trace # --reload
```

# Test
```bash
$ pip install pytest pytest-asyncio
$ python -m pytest -s -v
```

# Plan
* Add logger
