# TryTune

Heterogeneous System ML Pipeline Scheduling Framework with Triton Inference Server as Backend

## Install & Run
```bash
$ git clone https://github.com/cmpark0126/trytune-py
$ cd trytune-py
$ python -m pip install -r requirements.txt
$ python -m pip install tritonclient[http] # or tritonclient\[http\]
$ python -m pip install . # install trytune
$ uvicorn trytune.main:app --host 0.0.0.0 --port 80 --log-level trace # --reload if necessary
```

## Test
If you want to test basic functions of this framework, please use below:
```bash
$ python -m pytest -s -v -k "not k8s" # run with mock
...
collected 7 items / 1 deselected / 6 selected

tests/routers/test_modules.py::test_modules_scenario PASSED
tests/routers/test_scheduler.py::test_scheduler_scenario PASSED
tests/schemas/test_common.py::test_infer_schema PASSED
tests/schemas/test_module.py::test_add_module_schema PASSED
tests/schemas/test_pipeline.py::test_pipeline_add_schema PASSED
tests/schemas/test_scheduler.py::test_set_scheduler_schema PASSED
```

If you want to test real triton server behavior, please follow below:
```bash
$ kubectl get ingress # To check ingress address (without http maybe)
$ vi tests/routers/conftest.py # modify add_module_schema function with appropriate value
$ python -m pytest -s -v -k "k8s"
...
collected 7 items / 6 deselected / 1 selected 

tests/routers/test_module.py::test_module_scenario_on_k8s PASSED
```
