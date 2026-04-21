import json
import logging
import os
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.chatbot import FASTChatbot

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = FASTChatbot()


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found!</h1> Make sure it is in the root chatbot folder.")


@app.post("/chat")
def chat(req: ChatRequest):
    def generate():
        for chunk in bot.stream_response(req.message):
            if chunk:
                yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result_message = bot.process_pdf(temp_file_path)
        os.remove(temp_file_path)
        return {"message": result_message}
    except Exception as e:
        return {"message": f"Upload failed: {str(e)}"}
