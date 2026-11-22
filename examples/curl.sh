#!/usr/bin/env bash
# 先启动：python -m memex.http --port 8901
BASE=http://127.0.0.1:8901

curl -s -X POST "$BASE/remember" -H 'content-type: application/json' \
  -d '{"text":"Alice 偏好简短直接的消息。","scope":"facts","kind":"preference","source":"manual"}'

curl -s -X POST "$BASE/recall" -H 'content-type: application/json' \
  -d '{"q":"Alice 喜欢什么样的消息","k":3}'

curl -s "$BASE/core?budget=1200"
