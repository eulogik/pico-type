#!/usr/bin/env bash
# Promo demo for pico-type. Recorded with:
#   asciinema rec -c "bash promo/demo.sh" promo/picotype-demo.cast
export TERM="${TERM:-xterm-256color}"
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

run() {
  echo -e "\033[1;32m\$ $1\033[0m"
  sleep 0.6
  eval "$1"
  sleep 1.1
}

clear
echo -e "\033[1;34mpico-type — tiny byte-level content classifier (1.5M params, 7 heads, <6ms CPU)\033[0m"
sleep 1.2

run "picotype -t 'def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)' -p"
run "picotype -t '{\"name\": \"pico-type\", \"version\": \"0.2.0\", \"license\": \"apache-2.0\"}' -p"
run "picotype -t 'Bonjour le monde ! Comment allez-vous aujourd hui ?' -p"
run "picotype -t 'नमस्ते दुनिया! आप कैसे हैं?' -p"
run "picotype -t 'SELECT id, name, email FROM users WHERE active = 1 ORDER BY name;' -p"
run "picotype -t 'Traceback (most recent call last):\n  File \"app.py\", line 5, in <module>\n    main()\nZeroDivisionError: division by zero' -p"
run "picotype -t 'AKIAIOSFODNN7EXAMPLE1234567890abcdef' -p"
printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' > /tmp/picotype_png.bin
run "picotype --file /tmp/picotype_png.bin -p"

echo -e "\033[1;34mSame input, four Matryoshka tiers (tiny → pro):\033[0m"
sleep 1.0
for t in tiny small base pro; do
  echo -e "\033[1;32m\$ picotype -t 'package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"hi\") }' --tier $t\033[0m"
  sleep 0.4
  picotype -t 'package main

import "fmt"

func main() { fmt.Println("hi") }' --tier "$t" --pretty | python -c "import sys,json; d=json.load(sys.stdin); print('  coarse=%s  code_lang=%s  text_lang=%s' % (d['coarse']['label'], d['code_lang']['label'], d['text_lang']['label']))"
  sleep 0.7
done

echo
echo -e "\033[1;34mpico-type v0.2.0 — pip install pico-type\033[0m"
