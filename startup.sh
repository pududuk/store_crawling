#!/bin/sh
.venv/bin/gunicorn -w 1 -b 0.0.0.0:8001 app:app