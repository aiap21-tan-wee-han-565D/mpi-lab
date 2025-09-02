#!/usr/bin/env bash
set -e

# If key material is mounted, install it for mpiuser
if [ -d /sshkeys ]; then
  install -d -m 700 -o mpiuser -g mpiuser /home/mpiuser/.ssh
  if [ -f /sshkeys/id_ed25519 ]; then
    install -m 600 -o mpiuser -g mpiuser /sshkeys/id_ed25519 /home/mpiuser/.ssh/id_ed25519
  fi
  if [ -f /sshkeys/id_ed25519.pub ]; then
    install -m 644 -o mpiuser -g mpiuser /sshkeys/id_ed25519.pub /home/mpiuser/.ssh/id_ed25519.pub
    cat /sshkeys/id_ed25519.pub >> /home/mpiuser/.ssh/authorized_keys
    chown mpiuser:mpiuser /home/mpiuser/.ssh/authorized_keys
    chmod 600 /home/mpiuser/.ssh/authorized_keys
  fi
fi

# Give mpiuser ownership of the shared workspace if present
[ -d /work ] && chown -R mpiuser:mpiuser /work || true

# Start SSHD (needed because Open MPI spawns via SSH across nodes)
mkdir -p /var/run/sshd
/usr/sbin/sshd

# Keep the container alive
tail -f /dev/null
