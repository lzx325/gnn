#!/bin/bash
from_dir="$(pwd)/" # the last slash should not be forgot
remote="liz0f@10.73.106.192"
to_dir_basename="~/gnn"
to_dir="${remote}:${to_dir_basename}"
if [ "$1" = "-c" ]; then
    confirm="true"
else
    confirm=0
fi
include_options="--include=\"*.py\""
exclude_options="--exclude=\".vscode/\" --exclude=\"sync.sh\" --exclude='*.swp' --exclude=\"__pycache__\" --exclude='Data' --exclude='build' --exclude='dist' --exclude='gnn.egg-info'"
sync(){
    sync_command="rsync -rltD -vzhe ssh --progress ${include_options} ${exclude_options} ${1} ${2}"
    sync_command_dry_run="$sync_command --dry-run"
    if [ "$3" = "true" ]; then
        dry_run_output="$(eval $sync_command_dry_run)" || { code="$?";echo "rsync exited with status $code" 1>&2; return 1; } # use eval here because the command contains ""
        echo "$dry_run_output"
        read -p "confirm? [y]" choice
    else
        choice="y"
    fi

    if [ "$choice" = "y" ]; then
        eval "$sync_command" || { echo "rsync exited with status $?" 1>&2; return 1; }
    else
        echo "user cancelled syncing" 1>&2 ; return 1;
    fi
}
sync "$from_dir" "$to_dir" "$confirm" 
exit $?