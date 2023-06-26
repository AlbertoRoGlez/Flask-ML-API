from locust import HttpUser, between, task


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    @task(1)
    def index(self):
        self.client.get("/")
    
    @task(3)
    def predict(self):
        # Assuming that the predict endpoint requires an image file to be uploaded
        files = [("file", ("dog.jpeg", open("stress_test/dog.jpeg", "rb"), "image/jpeg"))]
        headers = {}
        payload = {}
        self.client.post(
            "http://localhost/predict",
            headers=headers,
            data=payload,
            files=files
        )