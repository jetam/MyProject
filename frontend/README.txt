
access frontend on localhost: file:///home/matej/code/MyProject/frontend/main.html

run a http server in project dir (MyProject): python3 -m http.server 3000  // this is static file server!
        HTTP SERVER RUNNING ON PORT 3000

server.py must be in MyProject so it sees frontend htmls

for fastapi you need server: python3 -m uvicorn server:app --reload --port 5000
        API server running on 5000
lsof -i -P -n | grep LISTEN // see which ports the server is listening on

connect with http: http://localhost:3000/frontend/composer.html

test api: http://127.0.0.1:5000/api/test




Frontend (JS fetch)
        ↓
Python API (FastAPI)
        ↓
C++ logic (your core)
        ↓
Response → back to frontend