#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Batch processing script for DeepReefMap
# Processes coral transect videos one-by-one, pulling from RCP and pushing
# results back, to minimize local disk usage.
#
# Usage:
#   ./process_transects.sh --video 2024/site_A/GX010297.MP4
#   ./process_transects.sh --folder 2024/site_A
# =============================================================================

# --- Configuration -----------------------------------------------------------
RCP_HOST="rcp"
RCP_BASE="/mnt/eceo/scratch/datasets/coral"
CSV_FILE="transects.csv"        # relative to RCP_BASE
FPS=5
LOCAL_WORK_DIR="./work"
LOCAL_OUTPUT_DIR="./output"
LOG_FILE="process_transects.log"

# Path to the project root (where src/reconstruct.py lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

DRY_RUN=false
VIDEO_PATH=""   # single video, relative to RCP_BASE
FOLDER_PATH=""  # folder to scan recursively, relative to RCP_BASE

# --- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --csv)       CSV_FILE="$2"; shift 2 ;;
        --fps)       FPS="$2"; shift 2 ;;
        --video)     VIDEO_PATH="$2"; shift 2 ;;
        --folder)    FOLDER_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--csv RCP_PATH] [--fps N] (--video RCP_PATH | --folder RCP_PATH)"
            echo ""
            echo "  --dry-run          Print commands without executing"
            echo "  --csv RCP_PATH     Metadata CSV on RCP, relative to RCP_BASE (default: transects.csv)"
            echo "  --fps N            Frames per second (default: 10)"
            echo "  --video RCP_PATH   Single video to process, relative to RCP_BASE"
            echo "  --folder RCP_PATH  Folder to scan recursively for videos, relative to RCP_BASE"
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

# Convert MM:SS to total seconds; passes "end" through as-is
mm_ss_to_seconds() {
    local ts="$1"
    if [[ "$ts" == "end" ]]; then
        echo "end"
        return
    fi
    local min="${ts%%:*}"
    local sec="${ts##*:}"
    echo $(( 10#$min * 60 + 10#$sec ))
}

# --- Validate ----------------------------------------------------------------
if [[ -z "$VIDEO_PATH" && -z "$FOLDER_PATH" ]]; then
    echo "ERROR: Must specify --video or --folder"
    exit 1
fi

if [[ -n "$VIDEO_PATH" && -n "$FOLDER_PATH" ]]; then
    echo "ERROR: Cannot specify both --video and --folder"
    exit 1
fi

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Make sure the .venv is set up in the project root."
    exit 1
fi

mkdir -p "$LOCAL_WORK_DIR" "$LOCAL_OUTPUT_DIR"

# --- Pull metadata CSV from RCP ----------------------------------------------
LOCAL_CSV="$SCRIPT_DIR/.transects_cache.csv"
remote_csv="$RCP_HOST:$RCP_BASE/$CSV_FILE"

log "Pulling metadata CSV from $remote_csv"
rsync -az "$remote_csv" "$LOCAL_CSV" || { echo "ERROR: Failed to pull CSV from $remote_csv"; exit 1; }

# --- Build video list --------------------------------------------------------
declare -a VIDEOS=()

if [[ -n "$VIDEO_PATH" ]]; then
    VIDEOS+=("$VIDEO_PATH")
else
    log "Scanning $RCP_HOST:$RCP_BASE/$FOLDER_PATH for videos (recursive)"
    while IFS= read -r abs_path; do
        # Strip RCP_BASE prefix + leading slash to get path relative to RCP_BASE
        rel_path="${abs_path#"$RCP_BASE"/}"
        VIDEOS+=("$rel_path")
    done < <(ssh "$RCP_HOST" "find '$RCP_BASE/$FOLDER_PATH' -type f -iname '*.mp4' ! -name '._*'" | sort)
fi

log "Found ${#VIDEOS[@]} video(s) to process"

# --- Pre-parse CSV into a flat file (one \x01-delimited row per line) --------
# Fields: filename,fw/bw,upside_down,transects,cut,comment,place,
#         transect_quality,transect_id,date,transect length,depth
PARSED_CSV=$(mktemp)
python3 -c "
import csv, sys
with open(sys.argv[1]) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if not row or row[0].startswith('#'):
            continue
        row += [''] * (12 - len(row))
        print('\x01'.join(row[:12]))
" "$LOCAL_CSV" > "$PARSED_CSV"
trap 'rm -f "$PARSED_CSV"' EXIT

# --- Process a single CSV row ------------------------------------------------
process_row() {
    local filename="$1"
    local direction="$2"
    local transects="$4"
    local comment="$6"
    local place="$7"
    local transect_quality="$8"
    local transect_id="$9"
    local date_field="${10}"
    local transect_length="${11}"

    if [[ -z "$transects" ]]; then
        log "  SKIP (no transects defined in CSV)"
        return
    fi

    log "  direction=$direction, transects=$transects, id=$transect_id"

    gopro_name="$(basename "${filename%.*}")"

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

        # --- Run reconstruct.py ----------------------------------------------
        # Run from src/ so default relative paths (e.g. ../example_inputs/) resolve
        run bash -c "cd '$SCRIPT_DIR/src' && '$PYTHON' reconstruct.py \
            --input_video='$local_video_abs' \
            --timestamp='$timestamp' \
            --out_dir='$out_dir_local_abs' \
            --fps='$FPS'"

        # --- Export results to CSV -------------------------------------------
        if [[ -n "$transect_id" ]]; then
            local_results_csv="$SCRIPT_DIR/results.csv"
            remote_results_csv="$RCP_BASE/results.csv"

            log "  Pulling results.csv from RCP (if exists)"
            if ! $DRY_RUN; then
                rsync -az "$RCP_HOST:$remote_results_csv" "$local_results_csv" 2>/dev/null \
                    && log "  results.csv pulled from RCP" \
                    || log "  results.csv not on RCP yet, will create fresh"
            else
                echo "  [DRY-RUN] rsync -az $RCP_HOST:$remote_results_csv $local_results_csv"
            fi

            log "  Exporting transect $transect_id to results.csv"
            run "$PYTHON" "$SCRIPT_DIR/src/export_to_csv.py" \
                --out_dir="$out_dir_local_abs" \
                --transect_id="$transect_id" \
                --video_name="$gopro_name" \
                --video_filename="$filename" \
                --transect_start="$start_ts" \
                --transect_end="$end_ts" \
                --date="$date_field" \
                --place="$place" \
                --transect_quality="$transect_quality" \
                --length="$transect_length" \
                --results_csv="$local_results_csv"

            log "  Uploading results.csv to RCP"
            run rsync -az "$local_results_csv" "$RCP_HOST:$remote_results_csv"
        else
            log "  SKIP export (no transect_id for $out_name)"
        fi

        # --- 4. Push results to RCP ------------------------------------------
        log "  Uploading results to $RCP_HOST:$out_dir_remote"
        run ssh "$RCP_HOST" "mkdir -p '$out_dir_remote'"
        run rsync -avz "$out_dir_local_abs/" "$RCP_HOST:$out_dir_remote"

        range_idx=$((range_idx + 1))
    done

    # --- 5. Clean up downloaded video ----------------------------------------
    log "  Removing local video: $local_video"
    run rm -f "$local_video"
}

# --- Main loop ---------------------------------------------------------------
log "Starting batch processing (dry_run=$DRY_RUN, csv=$CSV_FILE, fps=$FPS)"

for video in "${VIDEOS[@]}"; do
    log "Processing: $video"

    matched=false

    while IFS=$'\x01' read -r filename direction upside_down transects cut comment place \
            transect_quality transect_id date_field transect_length depth <&3; do
        if [[ "$filename" == "$video" ]]; then
            matched=true
            process_row "$filename" "$direction" "$upside_down" "$transects" \
                "$cut" "$comment" "$place" "$transect_quality" "$transect_id" \
                "$date_field" "$transect_length" "$depth"
        fi
    done 3< "$PARSED_CSV"

    if ! $matched; then
        log "  WARN: No CSV rows found for $video — skipping"
    fi

    log "Done: $video"
    echo "---"
done

log "Batch processing complete."

# --- Clean up local output directory -----------------------------------------
log "Removing local output directory: $LOCAL_OUTPUT_DIR"
run rm -rf "$LOCAL_OUTPUT_DIR"
