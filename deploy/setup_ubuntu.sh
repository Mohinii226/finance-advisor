#!/usr/bin/env bash
# Basic hardening for a fresh Ubuntu server that already has Docker. Run once as a sudo-capable user (not root).
#   - firewall: only SSH, HTTP, HTTPS   - automatic security updates   - fail2ban against SSH password guessing
# NOTE: Docker publishes container ports around ufw. That is safe here because only Caddy publishes ports (80/443) and the
# database has no published port. Do not add `ports:` to the db service.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y ufw fail2ban unattended-upgrades
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 443/udp
sudo ufw --force enable
sudo systemctl enable --now fail2ban
sudo dpkg-reconfigure -f noninteractive unattended-upgrades
echo "Done. Also: use SSH keys and set 'PasswordAuthentication no' in /etc/ssh/sshd_config."
