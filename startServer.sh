#!/usr/bin/env bash

kill $(lsof -t -i:3000)
kill $(lsof -t -i:5000)


sleep 1


STATIC_PORT=3000
echo "Starting static file server on port $STATIC_PORT"
python3 -m http.server 3000 > server_static.log 2>&1 & # this is static file server!

DYNAMIC_PORT=5000
echo "Starting api server on port $DYNAMIC_PORT"
python3 -m uvicorn server:app --reload --port 5000 > server_api.log 2>&1 & # this is dynamic fastapi server!
