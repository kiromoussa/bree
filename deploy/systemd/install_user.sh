#!/bin/sh
# Install the BREE user units for the current user. No sudo. Safe to run again.
#   deploy/systemd/install_user.sh            install and enable the purge + doctor timers (pipeline stays off)
#   deploy/systemd/install_user.sh --pipeline also enable the pipeline unit
# One-time admin step (needs root, not done here): `sudo loginctl enable-linger $USER` so user units start at boot
# without a login, and chrony or systemd-timesyncd running so the clock is synchronised.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
mkdir -p "$DEST" "$HOME/.config/bree"
chmod 700 "$HOME/.config/bree"
for f in bree-shadow.service bree-purge.service bree-purge.timer bree-doctor.service bree-doctor.timer; do
    install -m 644 "$HERE/$f" "$DEST/$f"
done
if [ ! -e "$HOME/.config/bree/retention.yaml" ]; then
    install -m 600 "$HERE/../../configs/retention_example.yaml" "$HOME/.config/bree/retention.yaml"
    echo "wrote ~/.config/bree/retention.yaml: set root to your shadow output folder"
fi
systemctl --user daemon-reload
systemctl --user enable --now bree-purge.timer bree-doctor.timer
[ "${1:-}" = "--pipeline" ] && systemctl --user enable bree-shadow.service
echo "installed. start the pipeline with: systemctl --user start bree-shadow ; logs: journalctl --user -u bree-shadow -f"
