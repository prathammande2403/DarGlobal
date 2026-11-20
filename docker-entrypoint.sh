#!/bin/sh
set -eu

# The UI container can start independently of the API's persisted Chroma index.
# The API owns retrieval and will report availability when a chat request arrives.
exec "$@"
