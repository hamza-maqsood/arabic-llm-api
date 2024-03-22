#!/bin/bash

quantize=${quantize:-$QUANTIZE}
sharded=${sharded:-$SHARDED}
max_input_length=${max_input_length:-$MAX_INPUT_LENGTH}
max_total_tokens=${max_total_tokens:-$MAX_TOTAL_TOKENS}


# Use command-line arguments if provided, else use environment variables
model_id=${model_id:-$MODEL_ID}
revision=${revision:-$MODEL_REVISION}
port=${port:-$PORT}
hostname=${hostname:-$HOSTNAME}

while getopts ":m:r:p:h:" opt; do
  case $opt in
    m) model_id="$OPTARG";;
    r) revision="$OPTARG";;
    p) port="$OPTARG";;
    h) hostname="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Checking if all required parameters are provided.
if [[ -z $model_id || -z $port || -z $hostname ]]; then
  echo "Usage: $0 -m MODEL_ID -p PORT -h HOSTNAME [-r REVISION]" >&2
  exit 1
fi

text-generation-launcher \
  --model-id "$model_id" \
  --quantize "$quantize" \
  --sharded "$sharded" \
  --max-input-length "$max_input_length" \
  --max-total-tokens "$max_total_tokens" \
  ${revision:+--revision "$revision"} \
  --port "$port" \
  --hostname "$hostname"
