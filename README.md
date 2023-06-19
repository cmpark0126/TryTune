# TryTune

Heterogeneous System ML Pipeline Scheduling Framework with Triton Inference Server as Backend

## Install & Run
```bash
$ git clone https://github.com/cmpark0126/trytune-py
$ cd trytune-py
$ mkdir .venv
$ python -m venv .venv/
$ source .venv/bin/activate
$ python -m pip install -r requirements.txt
$ python -m pip install tritonclient[http] # or tritonclient\[http\]
$ python -m pip install . # install trytune, -e if necessary
$ uvicorn trytune.main:app --host 0.0.0.0 --port 80 --log-level trace # --reload if necessary
```

## Test
If you want to test basic functions of this framework, please use below:
```bash
$ python -m pytest -s -v -k "not k8s" # run with mock
...
collected 13 items / 3 deselected / 10 selected

tests/routers/test_bls.py::test_bls_scenario Using cache found in /Users/chunmyong.park/.cache/torch/hub/pytorch_vision_v0.10.0
>> Result Top 5: [ 834: 8.05128002166748 233: 7.0492658615112305 416: 6.331830978393555 399: 6.245473384857178 842: 6.1228413581848145 ] << 
>> Result Top 5: [ 739: 7.768796920776367 982: 7.642239570617676 843: 7.566281795501709 713: 7.194638729095459 399: 6.773955345153809 ] << 
>> Result Top 5: [ 851: 5.9360737800598145 608: 4.957633018493652 617: 4.905662536621094 678: 4.8122878074646 395: 4.62005615234375 ] << 
>> Result Top 5: [ 399: 5.720999717712402 834: 5.176196098327637 416: 4.626550197601318 617: 4.484735488891602 916: 4.302403926849365 ] << 
PASSED
tests/routers/test_modules.py::test_modules_scenario PASSED
tests/routers/test_modules.py::test_builtin_modules_scenario 
>> Result is visualized at ./assets/FudanPed00054_result.png << 
>> Result is cropped at ./assets/FudanPed00054_person_{ 0 , 1 , 2 , 3 , .png } << 
PASSED
tests/routers/test_pipelines.py::test_pipelines_scenario 
>> Result is cropped at ./assets/FudanPed00054_person_{ 0 , 1 , 2 , 3 , .png } << 
PASSED
tests/routers/test_scheduler.py::test_scheduler_scenario PASSED
tests/schemas/test_common.py::test_infer_schema PASSED
tests/schemas/test_module.py::test_add_module_schema PASSED
tests/schemas/test_pipeline.py::test_pipeline_add_schema PASSED
tests/schemas/test_scheduler.py::test_set_scheduler_schema PASSED
tests/services/test_modules.py::test_builtin_resnet50_from_torch_hub Using cache found in /Users/chunmyong.park/.cache/torch/hub/pytorch_vision_v0.10.0
>> Result Top 5: [ 90: 12.474465370178223 92: 11.525705337524414 14: 9.660508155822754 136: 8.40636157989502 11: 8.22025203704834 ] << PASSED
```

If you want to test the triton inference server launched at the localhost, please follow below:
```bash
# Please follow the guideline of https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch to launch triton inference server on your localhost
$ vi tests/routers/conftest.py # modify add_module_schema function like below
{
    "name": "resnet50",
    "type": "triton",
    "urls": {
        "localhost": "http://localhost:8000"
    },
}
$ python -m pytest -s -v -k "k8s"
...
collected 7 items / 6 deselected / 1 selected 

tests/routers/test_modules.py::test_modules_scenario_on_k8s >> Result Top 5: [ 90: 12.474466323852539 92: 11.525705337524414 14: 9.660507202148438 136: 8.406360626220703 11: 8.220252990722656 ] << PASSED
```

If you want to test with your triton inference server deployed at k8s, please follow below:
```bash
$ kubectl get ingress (or kubectl get services) # To check address
# Assume that the ingress (or service) is connected to the triton inference server load resnet50 provided by https://github.com/triton-inference-server/tutorials/tree/main/Quick_Deploy/PyTorch
$ vi tests/routers/conftest.py # modify add_module_schema function with appropriate value
$ python -m pytest -s -v -k "k8s"
...
collected 7 items / 6 deselected / 1 selected 

tests/routers/test_modules.py::test_modules_scenario_on_k8s >> Result Top 5: [ 90: 12.474466323852539 92: 11.525705337524414 14: 9.660507202148438 136: 8.406360626220703 11: 8.220252990722656 ] << PASSED
```

## Contribution
```
$ pip install pre-commit
```
