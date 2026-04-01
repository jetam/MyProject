
access frontend on localhost: file:///home/matej/code/MyProject/frontend/main.html

run a http server in project dir (MyProject): python3 -m http.server
connect with http: http://localhost:8000/frontend/composer.html




Frontend (JS fetch)
        ↓
Python API (FastAPI)
        ↓
C++ logic (your core)
        ↓
Response → back to frontend