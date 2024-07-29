# SensoGPT: Remote Sensing Task with Chat GPT
Introduction
----
SensoGPT is an innovative solution that integrates ChatGPT with AI-based remote sensing models to automate complex interpretation tasks. By understanding user requests and performing precise task planning, it executes subtasks iteratively to generate comprehensive results.


### Usage
->Clone the repository "git clone https://github.com/Husky-AI9/SensoGPT"

->Change diretory into the backend

->Run "npm install"

->Run "npm run dev"

->Change directory into the frontend

->Run "pip install -r requirements.txt"

-> Dowload [weight(Google)](https://drive.google.com/file/d/165jeD0oi6fSpvWrpgfVBbzUOsyHN0xEq/view?usp=drive_link) and put it into the checkpoints folder

->Open main.py in the backend folder

->Replace GEMINI_API_KEY in main.py with your GEMINI key

->Close the file and in the same directory run "uvicorn main:app"

->Open your browser and enter localhost:3000 and start using the application
