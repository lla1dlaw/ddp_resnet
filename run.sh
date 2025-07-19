#!/bin/bash

cd ./logs/
FILE_TO_WAIT_FOR='output.log'
TIMEOUT_SECONDS=60 # Optional: set a timeout

rm -r ./*
cd ..
sbatch run.sbatch
echo "Waiting for $FILE_TO_WAIT_FOR to be created..."

# Loop until the file exists or timeout is reached
start_time=$(date +%s)
until [ -f "./logs/$FILE_TO_WAIT_FOR" ] || (($(date +%s) - start_time >= TIMEOUT_SECONDS)); do
  sleep 1 # Check every second
done

if [ -f "./logs/$FILE_TO_WAIT_FOR" ]; then
  echo "./logs/$FILE_TO_WAIT_FOR has been created."
  tail -f -n 100 ./logs/$FILE_TO_WAIT_FOR
else
  echo "Timeout: ./logs/$FILE_TO_WAIT_FOR was not created within $TIMEOUT_SECONDS seconds."
  exit 1 # Exit with an error code
fi
