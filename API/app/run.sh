#!/bin/bash
cd /app/app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
