# import pytest
# import respx
# from httpx import Response
# from fastapi import FastAPI
# from fastapi.testclient import TestClient
# from trytune.routers import schedulers


# @pytest.fixture
# def client() -> TestClient:
#     app = FastAPI()

#     # To test the router, you need to include it in the app.
#     app.include_router(schedulers.router)
#     return TestClient(app)


# @respx.mock
# def test_set_scheduler():
#     response = client.post(
#         "/schedulers/set",
#         json=SetSchedulerSchema(name="test_scheduler", config={"param": "value"}).dict(),
#     )
#     assert response.status_code == 200
#     assert response.json() == {
#         "message": "Scheduler set",
#         "name": "test_scheduler",
#         "config": {"param": "value"},
#     }


# @respx.mock
# def test_get_scheduler_metadata():
#     response = client.get("/schedulers/metadata")
#     assert response.status_code == 200
#     # Here, you will need to replace this assertion with the actual expected metadata
#     assert response.json() == {"expected_metadata_key": "expected_metadata_value"}


# @respx.mock
# def test_delete_scheduler():
#     response = client.delete("/schedulers/delete")
#     assert response.status_code == 200
#     assert response.json() == {"message": "Scheduler deleted"}
