from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file (if present)
load_dotenv()

app = FastAPI()

# Initialize OpenAI client using the API key from environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# This code defines a data model and an API endpoint using FastAPI and OpenAI's API.

class UserRequest(BaseModel):
    response: str

# Define a POST endpoint at '/ask_openai'.
# This endpoint expects a request body matching UserRequest.
# It sends the user's input to OpenAI's GPT-3.5-turbo model and returns the response.
@app.post("/ask_openai")
async def ask_openai(user_request: UserRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_request.response}],
        )
        # Extract the model's reply from the API response.
        answer = completion.choices[0].message.content
        # Return the reply in a JSON object with the key 'openai_response'.
        return {"openai_response": answer}
    except Exception as e:
        # If there's an error (e.g., network, API), raise an HTTP 500 error with details.
        raise HTTPException(status_code=500, detail=str(e))
