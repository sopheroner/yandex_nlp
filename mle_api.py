from fastapi import FastAPI, Body
from handler import FastApiHandler


app = FastAPI()
app.handler = FastApiHandler()


@app.post("/api/mle")
def get_prediction_for_item(user_id: str, model_params: dict):
    all_params = {"user_id": user_id,
                   "model_params": model_params}
    return app.handler.handle(all_params)
