#!/usr/bin/env bash

# Static server: for webpage

kill $(lsof -t -i:3000)


> server_static.log


STATIC_PORT=3000
echo "Starting static file server on port $STATIC_PORT"
PYTHONUNBUFFERED=1 python3 -m http.server 3000 | tee server_static.log

