from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Badminton Footwork Detection API is running!"}