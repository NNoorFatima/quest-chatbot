from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from app.chatbot import FASTChatbot
from fastapi.responses import HTMLResponse

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

class ChatResponse(BaseModel):
    reply: str

# @app.get("/")
@app.get("/", response_class=HTMLResponse)
def home():
    # This reads your index.html file and serves it directly
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found!</h1> Make sure it is in the root chatbot folder.")
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        print("Received message:", req.message)
        response = bot.get_response(req.message)
        return {"reply": response}
    except Exception as e:
        return {"reply": "Sorry, something went wrong. Please try again."}

#Upload PDF Endpoint ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Create a temporary path to save the uploaded file
        temp_file_path = f"temp_{file.filename}"
        
        # Save the file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Tell the chatbot to process the PDF
        result_message = bot.process_pdf(temp_file_path)
        
        # Clean up: delete the temporary file now that it's in the vector store
        os.remove(temp_file_path)
        
        return {"message": result_message}
    except Exception as e:
        return {"message": f"Upload failed: {str(e)}"}