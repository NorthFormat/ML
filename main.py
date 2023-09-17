from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from Mlpart import MLpart

app = FastAPI()
model = MLpart()


class Text(BaseModel):
    text: str
    grammatic: Optional[bool] = False
    paragraph: Optional[bool] = False
    foramt: Optional[bool] = False


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chagetext/")
async def flex(data: Text):
    return model.do(data.text, data.grammatic, data.paragraph, data.foramt)
