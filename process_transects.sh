#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Batch processing script for DeepReefMap
# Processes coral transect videos one-by-one, pulling from RCP and pushing
# results back, to minimize local disk usage.
# =============================================================================

# --- Configuration -----------------------------------------------------------
RCP_HOST="rcp"
RCP_BASE="/mnt/eceo/scratch/datasets/coral/2025_10_eritrea/TRSC_er_102025_data/TRSC_er_10_2025_3D"
CSV_FILE="transects.csv"
FPS=10
LOCAL_WORK_DIR="./work"
LOCAL_OUTPUT_DIR="./output"
LOG_FILE="process_transects.log"

# Path to the project root (where src/reconstruct.py lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECONSTRUCT_PY="$SCRIPT_DIR/src/reconstruct.py"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

DRY_RUN=false

# --- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --csv)      CSV_FILE="$2"; shift 2 ;;
        --fps)      FPS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--csv FILE] [--fps N]"
            echo ""
            echo "  --dry-run   Print commands without executing"
            echo "  --csv FILE  Path to transects CSV (default: transects.csv)"
            echo "  --fps N     Frames per second (default: 10)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helpers -----------------------------------------------------------------
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

run() {
    if $DRY_RUN; then
        echo "  [DRY-RUN] $*"
    else
        "$@"
    fi
}

# Convert MM:SS to total seconds
mm_ss_to_seconds() {
    local ts="$1"
    local min="${ts%%:*}"
    local sec="${ts##*:}"
    echo $(( 10#$min * 60 + 10#$sec ))
}

# --- Validate ----------------------------------------------------------------
if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Make sure the .venv is set up in the project root."
    exit 1
fi

mkdir -p "$LOCAL_WORK_DIR" "$LOCAL_OUTPUT_DIR"

# --- Main loop ---------------------------------------------------------------
log "Starting batch processing (dry_run=$DRY_RUN, csv=$CSV_FILE, fps=$FPS)"

# Read CSV, skip header
tail -n +2 "$CSV_FILE" | while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue

    # Use Python for reliable CSV parsing of this line
    # Fields: filename,fw/bw,upside_down,transects,cut,comment,place,
    #         transect_quality,transect_id,date,transect length,depth
    IFS=$'\t' read -r filename direction upside_down transects cut comment place \
        transect_quality transect_id date_field transect_length depth \
        < <(python3 -c "
import csv, io, sys
row = next(csv.reader(io.StringIO(sys.argv[1])))
row += [''] * (12 - len(row))
print('\t'.join(row[:12]))
" "$line")

    # Skip rows with empty transects
    if [[ -z "$transects" ]]; then
        log "SKIP (no transects): $filename"
        continue
    fi

    log "Processing: $filename (direction=$direction, transects=$transects, id=$transect_id)"

    # --- 1. Download video from RCP ------------------------------------------
    local_video="$LOCAL_WORK_DIR/$(basename "$filename")"
    remote_video="$RCP_HOST:$RCP_BASE/$filename"

    log "  Downloading $remote_video"
    run rsync -avz --progress "$remote_video" "$local_video"

    # --- 2. Build reverse flag -----------------------------------------------
    reverse_flag=""
    if [[ "$direction" == "bw" ]]; then
        reverse_flag="--reverse"
    fi

    # --- 3. Process each transect range --------------------------------------
    IFS=',' read -ra ranges <<< "$transects"
    range_idx=0

    for range in "${ranges[@]}"; do
        range=$(echo "$range" | xargs)  # trim whitespace
        [[ -z "$range" ]] && continue

        # Build output name from filename path: replace / with _ and drop extension
        out_name="${filename%.*}"
        out_name="${out_name//\//_}"
        if [[ ${#ranges[@]} -gt 1 ]]; then
            out_name="${out_name}_${range_idx}"
        fi

        out_dir_local="$LOCAL_OUTPUT_DIR/$out_name"
        out_dir_remote="$RCP_BASE/output/$out_name/"

        # Convert MM:SS-MM:SS to seconds
        start_ts="${range%%-*}"
        end_ts="${range##*-}"
        start_sec=$(mm_ss_to_seconds "$start_ts")
        end_sec=$(mm_ss_to_seconds "$end_ts")
        timestamp="${start_sec}-${end_sec}"

        log "  Transect $range_idx: range=$range (${timestamp}s), output=$out_name"

        local_video_abs="$(readlink -f "$local_video")"
        out_dir_local_abs="$(readlink -f "$LOCAL_OUTPUT_DIR")/$out_name"
        mkdir -p "$out_dir_local_abs"

        # --- Run reconstruct.py directly -------------------------------------
        # Run from src/ so default relative paths (e.g. ../example_inputs/) resolve
        run bash -c "cd '$SCRIPT_DIR/src' && '$PYTHON' reconstruct.py \
            --input_video='$local_video_abs' \
            --timestamp='$timestamp' \
            --out_dir='$out_dir_local_abs' \
            --fps='$FPS'\
            --render_video"

        # --- 4. Push results to RCP ------------------------------------------
        log "  Uploading results to $RCP_HOST:$out_dir_remote"
        run ssh "$RCP_HOST" "mkdir -p '$out_dir_remote'"
        run rsync -avz "$out_dir_local_abs/" "$RCP_HOST:$out_dir_remote"

        range_idx=$((range_idx + 1))
    done

    # --- 5. Clean up downloaded video ----------------------------------------
    log "  Removing local video: $local_video"
    run rm -f "$local_video"

    log "Done: $filename"
    echo "---"
done

log "Batch processing complete."
