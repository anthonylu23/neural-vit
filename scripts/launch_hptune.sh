#!/bin/bash
# Launch Vertex AI Hyperparameter Tuning Job
#
# Usage:
#   ./scripts/launch_hptune.sh [--build] [--dry-run]
#
# Options:
#   --build    Build and push Docker image before launching
#   --dry-run  Print the gcloud command without executing

set -e

PROJECT_ID="lfp-temporal-vit"
REGION="us-central1"
IMAGE_URI="us-central1-docker.pkg.dev/${PROJECT_ID}/vertex-job1/temporal-vit:latest"
CONFIG_FILE="vertex_hptune_config.yaml"
DISPLAY_NAME="temporal-vit-hptune-$(date +%Y%m%d-%H%M%S)"

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
    echo "Building Docker image..."
    docker build -t temporal-vit .
    
    echo "Tagging image..."
    docker tag temporal-vit "${IMAGE_URI}"
    
    echo "Pushing image to Artifact Registry..."
    docker push "${IMAGE_URI}"
    
    echo "Image pushed successfully."
fi

# Verify config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Build the gcloud command
GCLOUD_CMD="gcloud ai hp-tuning-jobs create \
    --region=${REGION} \
    --display-name=${DISPLAY_NAME} \
    --config=${CONFIG_FILE} \
    --project=${PROJECT_ID}"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run - would execute:"
    echo "${GCLOUD_CMD}"
else
    echo "Launching HP tuning job: ${DISPLAY_NAME}"
    eval "${GCLOUD_CMD}"
    echo ""
    echo "Job submitted successfully!"
    echo "Monitor at: https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs?project=${PROJECT_ID}"
fi
