def test_scheduler_scenario(client) -> None:  # type: ignore
    scheduler_schema = {"name": "fifo", "config": {}}
    response = client.post(f"/scheduler/set", json=scheduler_schema)
    assert response.status_code == 200, response.content
    assert response.json() == {"message": "Scheduler set", "name": "fifo", "config": {}}

    response = client.get("/scheduler/metadata")
    assert response.status_code == 200, response.content
    assert response.json() == {"name": "fifo", "config": {}}

    response = client.delete("/scheduler/delete")
    assert response.status_code == 200, response.content
    assert response.json() == {"message": "Scheduler deleted"}
