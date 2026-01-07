#!/bin/bash
# Launch Vertex AI Training Job (single run)
#
# Usage:
#   ./scripts/launch_train.sh [--build] [--dry-run]
#
# Options:
#   --build    Build and push Docker image before launching
#   --dry-run  Print the gcloud command without executing
#
# Examples:
#   ./scripts/launch_train.sh --build          # Build image and launch job
#   ./scripts/launch_train.sh                  # Launch job (image already pushed)
#   ./scripts/launch_train.sh --dry-run        # Preview the command

set -e

PROJECT_ID="lfp-temporal-vit"
REGION="us-central1"
IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/vertex-job1/temporal-vit:latest"
CONFIG_FILE="vertex_custom_job_a100_tensorboard.yaml"
DISPLAY_NAME="temporal-vit-$(date +%Y%m%d-%H%M%S)"

BUILD=false
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --build)
            BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--build] [--dry-run]"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$(dirname "$0")/.."

# Build and push Docker image if requested
if [ "$BUILD" = true ]; then
    echo "=========================================="
    echo "Building Docker image..."
    echo "=========================================="
    docker build -t temporal-vit .
    
    echo ""
    echo "Tagging image..."
    docker tag temporal-vit "${IMAGE_URI}"
    
    echo ""
    echo "Pushing image to Artifact Registry..."
    docker push "${IMAGE_URI}"
    
    echo ""
    echo "Image pushed successfully."
    echo "=========================================="
fi

# Verify config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Build the gcloud command
GCLOUD_CMD="gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${DISPLAY_NAME} \
    --config=${CONFIG_FILE} \
    --project=${PROJECT_ID}"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - would execute:"
    echo ""
    echo "${GCLOUD_CMD}"
else
    echo ""
    echo "Launching training job: ${DISPLAY_NAME}"
    echo ""
    eval "${GCLOUD_CMD}"
    echo ""
    echo "=========================================="
    echo "Job submitted successfully!"
    echo "=========================================="
    echo ""
    echo "Monitor at:"
    echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
    echo ""
    echo "TensorBoard:"
    echo "  https://console.cloud.google.com/vertex-ai/experiments/tensorboard?project=${PROJECT_ID}"
fi
