#!/usr/bin/env bash
# assumes: python -m memex.http --port 8901
BASE=http://127.0.0.1:8901

curl -s -X POST "$BASE/remember" -H 'content-type: application/json' \
  -d '{"text":"Alice prefers short, direct messages.","scope":"facts","kind":"preference","source":"manual"}'

curl -s -X POST "$BASE/recall" -H 'content-type: application/json' \
  -d '{"q":"how does alice like messages","k":3}'

curl -s "$BASE/core?budget=1200"
