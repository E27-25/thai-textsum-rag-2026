#!/usr/bin/env bash
# swait.sh — estimate how long a SLURM sbatch job may wait before running.
# Usage: bash swait.sh <JOB_ID>

set -u

usage() {
  echo "Usage: bash swait.sh <JOB_ID>"
  echo "Example: bash swait.sh 5769690"
}

die() {
  echo "Error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

human_time() {
  mins="$1"

  if [ "$mins" -lt 0 ]; then
    echo "unknown"
  elif [ "$mins" -lt 60 ]; then
    echo "${mins} minute(s)"
  elif [ "$mins" -lt 1440 ]; then
    echo "$((mins / 60)) hour(s) $((mins % 60)) minute(s)"
  else
    echo "$((mins / 1440)) day(s) $(((mins % 1440) / 60)) hour(s)"
  fi
}

to_epoch() {
  date -d "$1" +%s 2>/dev/null || echo ""
}

estimate_from_slurm_start() {
  label="$1"
  start_time="$2"

  [ -n "$start_time" ] || return 1
  [ "$start_time" = "N/A" ] && return 1
  [ "$start_time" = "Unknown" ] && return 1
  [ "$start_time" = "None" ] && return 1

  now_epoch=$(date +%s)
  start_epoch=$(to_epoch "$start_time")

  [ -n "$start_epoch" ] || return 1

  mins=$(( (start_epoch - now_epoch + 59) / 60 ))
  [ "$mins" -lt 0 ] && mins=0

  echo "Estimated start : $start_time"
  echo "Estimated wait  : $(human_time "$mins")"
  echo "Method          : $label"
  exit 0
}

JOB_ID="${1:-}"

[ -n "$JOB_ID" ] || {
  usage
  exit 1
}

[[ "$JOB_ID" =~ ^[0-9]+$ ]] || die "Invalid job ID: $JOB_ID"

need_cmd squeue
need_cmd date
need_cmd awk

INFO=$(squeue -j "$JOB_ID" -h -o "%i|%T|%P|%u|%Q|%R|%D|%C|%m|%l" 2>/dev/null)

[ -n "$INFO" ] || die "Job $JOB_ID not found in squeue. It may have finished or the ID may be wrong."

IFS='|' read -r ID STATE PARTITION USER PRIORITY REASON NODES CPUS MEM_LIMIT TIME_LIMIT <<< "$INFO"

echo "Job ID          : $ID"
echo "State           : $STATE"
echo "Partition       : $PARTITION"
echo "User            : $USER"
echo "Priority        : $PRIORITY"
echo "Reason          : $REASON"
echo "Requested nodes : $NODES"
echo "Requested CPUs  : $CPUS"
echo "Requested memory: $MEM_LIMIT"
echo "Time limit      : $TIME_LIMIT"
echo

# Case 1: already running
if [ "$STATE" = "RUNNING" ]; then
  echo "Estimated wait  : 0 minute(s)"
  echo "Status          : already running"
  exit 0
fi

# Case 2: not pending
if [ "$STATE" != "PENDING" ]; then
  echo "Estimated wait  : unknown"
  echo "Status          : job is not pending; current state is $STATE"
  exit 0
fi

# Best estimate: ask SLURM directly through scontrol, if available
if command -v scontrol >/dev/null 2>&1; then
  JOB_DETAIL=$(scontrol show job "$JOB_ID" 2>/dev/null)

  START_TIME=$(echo "$JOB_DETAIL" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^StartTime=/) {
          sub(/^StartTime=/, "", $i)
          print $i
          exit
        }
      }
    }
  ')

  SUBMIT_TIME=$(echo "$JOB_DETAIL" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^SubmitTime=/) {
          sub(/^SubmitTime=/, "", $i)
          print $i
          exit
        }
      }
    }
  ')

  ELIGIBLE_TIME=$(echo "$JOB_DETAIL" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^EligibleTime=/) {
          sub(/^EligibleTime=/, "", $i)
          print $i
          exit
        }
      }
    }
  ')

  echo "Submit time     : ${SUBMIT_TIME:-unknown}"
  echo "Eligible time   : ${ELIGIBLE_TIME:-unknown}"
  echo "SLURM StartTime : ${START_TIME:-unknown}"
  echo

  estimate_from_slurm_start "scontrol StartTime" "$START_TIME"
fi

# Second-best estimate: squeue --start, if backfill prediction is available
START_BY_SQUEUE=$(squeue --start -j "$JOB_ID" -h -o "%S" 2>/dev/null | head -n 1)
estimate_from_slurm_start "squeue --start" "$START_BY_SQUEUE"

# Fallback 1: queue position by priority within the same partition
JOBS_AHEAD=$(
  squeue -h -t PENDING -p "$PARTITION" -o "%Q|%i" 2>/dev/null |
  awk -F'|' -v myprio="$PRIORITY" -v myid="$JOB_ID" '
    {
      prio = $1 + 0
      jid = $2 + 0
      if (prio > myprio || (prio == myprio && jid < myid)) ahead++
    }
    END { print ahead + 0 }
  '
)

# Fallback 2: estimate using currently running jobs in the partition
# We look at running jobs and their remaining walltime.
# This is a rough proxy for when resources may become free.
RESOURCE_EST=$(
  squeue -h -t RUNNING -p "$PARTITION" -o "%L" 2>/dev/null |
  awk '
    function to_min(t, d,h,m,s,a,n) {
      d = 0
      if (t ~ /-/) {
        split(t, a, "-")
        d = a[1]
        t = a[2]
      }

      n = split(t, a, ":")
      if (n == 3) {
        h = a[1]; m = a[2]; s = a[3]
      } else if (n == 2) {
        h = 0; m = a[1]; s = a[2]
      } else {
        h = 0; m = t; s = 0
      }

      return d * 1440 + h * 60 + m + int((s + 59) / 60)
    }

    $1 != "N/A" && $1 != "NOT_SET" && $1 != "UNLIMITED" {
      mins = to_min($1)
      if (mins >= 0) {
        a[++n] = mins
      }
    }

    END {
      if (n == 0) {
        print ""
        exit
      }

      # Sort manually for POSIX-ish awk compatibility.
      for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
          if (a[j] < a[i]) {
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp
          }
        }
      }

      # Use lower quartile remaining time as an optimistic resource-release estimate.
      idx = int(n * 0.25)
      if (idx < 1) idx = 1
      print a[idx]
    }
  '
)

# Fallback 3: recent historical wait time from sacct, if accounting is available
HIST_EST=""

if command -v sacct >/dev/null 2>&1; then
  SINCE=$(date -d "14 days ago" +%Y-%m-%d 2>/dev/null || date +%Y-%m-%d)

  HIST_EST=$(
    sacct -X -n -S "$SINCE" -p -o JobIDRaw,Partition,Submit,Start,State 2>/dev/null |
    awk -F'|' -v part="$PARTITION" '
      $2 == part && $3 != "" && $4 != "" && $4 != "Unknown" {
        print $3 "|" $4
      }
    ' |
    while IFS='|' read -r submit start; do
      s=$(date -d "$submit" +%s 2>/dev/null || true)
      r=$(date -d "$start" +%s 2>/dev/null || true)

      if [ -n "${s:-}" ] && [ -n "${r:-}" ] && [ "$r" -gt "$s" ]; then
        echo $(( (r - s) / 60 ))
      fi
    done |
    awk '
      {
        a[++n] = $1
      }

      END {
        if (n == 0) exit

        for (i = 1; i <= n; i++) {
          for (j = i + 1; j <= n; j++) {
            if (a[j] < a[i]) {
              tmp = a[i]
              a[i] = a[j]
              a[j] = tmp
            }
          }
        }

        mid = int((n + 1) / 2)
        print a[mid]
      }
    '
  )
fi

echo "SLURM exact estimate : unavailable"
echo "Jobs ahead           : $JOBS_AHEAD pending job(s) in partition '$PARTITION'"

if [ -n "$RESOURCE_EST" ]; then
  echo "Resource estimate    : about $(human_time "$RESOURCE_EST") until some running jobs release resources"
else
  echo "Resource estimate    : unavailable"
fi

if [ -n "$HIST_EST" ]; then
  echo "Historical median    : $(human_time "$HIST_EST") for recent jobs in this partition"
else
  echo "Historical median    : unavailable"
fi

echo

# Combine rough signals.
# This is intentionally simple and conservative.
if [ -n "$HIST_EST" ] && [ -n "$RESOURCE_EST" ]; then
  EST=$(( (HIST_EST + RESOURCE_EST) / 2 + JOBS_AHEAD * 3 ))
  echo "Estimated wait       : approximately $(human_time "$EST")"
  echo "Method               : history + running-job availability + queue position"
elif [ -n "$HIST_EST" ]; then
  EST=$(( HIST_EST + JOBS_AHEAD * 3 ))
  echo "Estimated wait       : approximately $(human_time "$EST")"
  echo "Method               : historical median + queue position"
elif [ -n "$RESOURCE_EST" ]; then
  EST=$(( RESOURCE_EST + JOBS_AHEAD * 3 ))
  echo "Estimated wait       : approximately $(human_time "$EST")"
  echo "Method               : running-job availability + queue position"
else
  echo "Estimated wait       : unknown"
  echo "Method               : not enough scheduler/accounting information"
fi

echo
echo "Note:"
echo "  This is an approximation. Exact start time is only available if SLURM"
echo "  publishes StartTime through 'scontrol show job' or 'squeue --start'."
echo "  Priority, fairshare, QoS, reservations, backfill, and job size can all"
echo "  change the actual start time."
