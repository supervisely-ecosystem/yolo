set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker build --no-cache \
  -f "${SCRIPT_DIR}/Dockerfile.deploy" \
  -t supervisely/yolo:1.0.8-deploy \
  "$PROJECT_ROOT"

docker push supervisely/yolo:1.0.8-deploy