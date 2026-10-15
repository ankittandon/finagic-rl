#!/usr/bin/env bash
# Run this ONCE in the terminal tab where you already exported your keys:
#   bash save_env.sh
# It writes the three env vars (incl. your Fireworks key) to .env, which is
# gitignored. Later steps are run with:  source .env && <command>
set -euo pipefail
: "${FIREWORKS_API_KEY:?FIREWORKS_API_KEY is not set in this shell}"
: "${FIREWORKS_ACCOUNT_ID:?FIREWORKS_ACCOUNT_ID is not set in this shell}"
: "${SEC_USER_AGENT:?SEC_USER_AGENT is not set in this shell}"
umask 077
{
  printf 'export SEC_USER_AGENT=%q\n' "$SEC_USER_AGENT"
  printf 'export FIREWORKS_ACCOUNT_ID=%q\n' "$FIREWORKS_ACCOUNT_ID"
  printf 'export FIREWORKS_API_KEY=%q\n' "$FIREWORKS_API_KEY"
} > .env
echo "wrote .env (key length: ${#FIREWORKS_API_KEY} chars)"
