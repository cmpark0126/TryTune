# TryTune

Heterogeneous System Model Pipeline Scheduling Framework with Triton Inference Server as Backend

# Install & Run
```bash
$ git clone https://github.com/cmpark0126/trytune-py
$ cd trytune-py
$ python -m pip install -r requirements.txt
$ python -m pip install . # install trytune
$ uvicorn fastapi.main:app --host 0.0.0.0 --port 80 --log-level trace # --reload if necessary
```

# Test
```bash
$ python -m pytest -s -v
```

# Plan
* Add logger
