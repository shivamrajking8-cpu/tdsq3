from fastapi import FastAPI, HTTPException
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

iris = load_iris()
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)

class_names = ["setosa", "versicolor", "virginica"]

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict")
async def predict(sl: float, sw: float, pl: float, pw: float):
    try:
        features = np.array([[sl, sw, pl, pw]])
        pred = int(model.predict(features)[0])
        return {
            "prediction": pred,
            "class_name": class_names[pred]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
