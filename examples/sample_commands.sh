#!/bin/bash

# RTVT-LipSync API Test Script
# This script walks through the complete process of uploading a video,
# translating it, and downloading the result.

set -e

# Settings - you can change these
API_URL="${API_URL:-http://localhost:8000}"
TEST_VIDEO="${TEST_VIDEO:-demo.mp4}"
TARGET_LANG="${TARGET_LANG:-es}"

# Pretty colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}Testing RTVT-LipSync API${NC}"
echo "=================================="

# First, let's check if the API is running
echo -e "${YELLOW}Checking if API is available...${NC}"
if ! curl -s -f "${API_URL}/health" > /dev/null; then
    echo "ERROR: Can't reach the API at ${API_URL}"
    echo "Make sure the server is running:"
    echo "  docker-compose up -d"
    echo "  OR"
    echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi
echo -e "${GREEN}API is running!${NC}"
echo

# Make sure we have a video file to work with
echo -e "${YELLOW}Checking for test video...${NC}"
if [ ! -f "${TEST_VIDEO}" ]; then
    echo "No test video found. Creating a simple one..."
    
    # Create a basic test video (5 seconds with text)
    ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 \
           -f lavfi -i sine=frequency=1000:duration=5 \
           -vf "drawtext=text='Test Video':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2" \
           -c:v libx264 -c:a aac -shortest "${TEST_VIDEO}" -y 2>/dev/null || {
        echo "ERROR: Couldn't create test video. Please provide one manually."
        exit 1
    }
    echo -e "${GREEN}Test video created!${NC}"
fi

# Step 1: Upload the video
echo -e "${YELLOW}Uploading video...${NC}"
upload_result=$(curl -s -X POST "${API_URL}/upload" -F "file=@${TEST_VIDEO}")
upload_id=$(echo "${upload_result}" | grep -o '"upload_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "${upload_id}" ]; then
    echo "Upload failed: ${upload_result}"
    exit 1
fi

echo -e "${GREEN}Video uploaded successfully!${NC}"
echo "Upload ID: ${upload_id}"
echo

# Step 2: Start the translation process
echo -e "${YELLOW}Starting translation to ${TARGET_LANG}...${NC}"
process_result=$(curl -s -X POST "${API_URL}/process" \
    -H "Content-Type: application/json" \
    -d "{\"upload_id\":\"${upload_id}\",\"target_lang\":\"${TARGET_LANG}\"}")

job_id=$(echo "${process_result}" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "${job_id}" ]; then
    echo "Failed to start processing: ${process_result}"
    exit 1
fi

echo -e "${GREEN}Processing started!${NC}"
echo "Job ID: ${job_id}"
echo

# Step 3: Wait for processing to complete
echo -e "${YELLOW}Waiting for processing to finish...${NC}"
echo "(This might take a few minutes)"

max_wait=600  # 10 minutes
waited=0
check_every=5

while [ ${waited} -lt ${max_wait} ]; do
    status_info=$(curl -s "${API_URL}/process/${job_id}/status")
    
    status=$(echo "${status_info}" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    progress=$(echo "${status_info}" | grep -o '"progress":[0-9]*' | cut -d':' -f2)
    message=$(echo "${status_info}" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
    step=$(echo "${status_info}" | grep -o '"step":"[^"]*"' | cut -d'"' -f4 | head -1)
    
    # Show current progress
    echo -ne "\r[${progress}%] ${step}: ${message}                    "
    
    if [ "${status}" = "completed" ]; then
        echo
        echo -e "${GREEN}Processing finished!${NC}"
        break
    elif [ "${status}" = "failed" ]; then
        echo
        echo "Processing failed: ${status_info}"
        exit 1
    fi
    
    sleep ${check_every}
    waited=$((waited + check_every))
done

if [ ${waited} -ge ${max_wait} ]; then
    echo
    echo "Timed out waiting for processing. Check manually with:"
    echo "curl ${API_URL}/process/${job_id}/status"
    exit 1
fi

# Step 4: Download the result
echo -e "${YELLOW}Downloading result...${NC}"
output_file="translated_${TARGET_LANG}_$(date +%Y%m%d_%H%M%S).mp4"

http_code=$(curl -s -w "%{http_code}" -o "${output_file}" "${API_URL}/media/${job_id}")

if [ "${http_code}" = "200" ]; then
    file_size=$(ls -lh "${output_file}" | awk '{print $5}')
    echo -e "${GREEN}Download complete!${NC}"
    echo "Saved as: ${output_file} (${file_size})"
    
    # Show basic video info if ffprobe is available
    if command -v ffprobe &> /dev/null; then
        echo "Video details:"
        ffprobe -v quiet -print_format json -show_format -show_streams "${output_file}" 2>/dev/null | \
            grep -E '"duration"|"width"|"height"' || true
    fi
else
    echo "Download failed (HTTP ${http_code})"
    rm -f "${output_file}"
    exit 1
fi

echo
echo -e "${GREEN}All done!${NC}"
echo "=================================="
echo "Summary:"
echo "  Upload ID: ${upload_id}"
echo "  Job ID: ${job_id}"
echo "  Result: ${output_file}"
echo
echo "To watch the video:"
echo "  ffplay ${output_file}          # with ffmpeg"
echo "  open ${output_file}            # on macOS"
echo "  xdg-open ${output_file}        # on Linux"

echo
echo "Other useful commands:"
echo "  curl ${API_URL}/jobs                    # list all jobs"
echo "  curl ${API_URL}/process/${job_id}/status    # check job status"
echo "  curl ${API_URL}/health                 # API health check"
echo "  open ${API_URL}/docs                   # API documentation"

echo
echo "For large files, you can use chunked upload:"
cat << 'CHUNKED_EXAMPLE'

# Split large file into 10MB chunks
split -b 10485760 big_video.mp4 chunk_

# Upload each chunk
upload_id=$(uuidgen)
total_chunks=$(ls chunk_* | wc -l)
index=0

for chunk in chunk_*; do
  curl -X POST "${API_URL}/upload/chunk" \
    -H "Upload-Id: $upload_id" \
    -H "Chunk-Index: $index" \
    -H "Total-Chunks: $total_chunks" \
    -H "Filename: big_video.mp4" \
    -F "file=@$chunk"
  index=$((index + 1))
done

# Clean up chunks
rm chunk_*

CHUNKED_EXAMPLE
