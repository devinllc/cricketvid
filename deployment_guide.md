# 🚀 Cricket Video Assessment System — Deployment Guide

## Step 1 — Test Docker Locally First

Make sure Docker Desktop is installed on your Mac, then:

```bash
cd /Users/rameshvishwakarma/cricketVideo

# Build the Docker image
docker build -t cricket-assessment:latest .

# Run it locally
docker compose up
```

Visit http://localhost:8000 — if it works locally, it will work on a server.

```bash
# Stop it
docker compose down
```

---

## Step 2 — Choose Your Server

### Recommended: DigitalOcean Droplet (~$12/month)

Best for POC — simple, cheap, good docs.

**Minimum specs:**
| Spec | Value |
|---|---|
| RAM | 2 GB (4 GB recommended for MediaPipe) |
| CPU | 2 vCPUs |
| Storage | 50 GB SSD |
| OS | Ubuntu 22.04 LTS |
| Region | Closest to your users |

**Create a Droplet:**
1. Go to https://cloud.digitalocean.com
2. Create → Droplets → Ubuntu 22.04
3. Choose **Basic → Regular → $12/mo (2GB RAM)**
4. Add your SSH key
5. Create Droplet — note the **IP address**

---

## Step 3 — Initial Server Setup

SSH into your server:

```bash
ssh root@YOUR_SERVER_IP
```

### Install Docker + Docker Compose

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Start Docker
systemctl enable docker
systemctl start docker

# Install Docker Compose plugin
apt install docker-compose-plugin -y

# Verify
docker --version
docker compose version
```

---

## Step 4 — Deploy the App

### Option A — Git Clone (Recommended)

```bash
# On your Mac — push to GitHub first
cd /Users/rameshvishwakarma/cricketVideo
git init
git add .
git commit -m "Initial POC"
git remote add origin https://github.com/YOUR_USERNAME/cricket-assessment.git
git push -u origin main
```

```bash
# On server
git clone https://github.com/YOUR_USERNAME/cricket-assessment.git
cd cricket-assessment
docker compose up -d --build
```

### Option B — Direct SCP Upload

```bash
# On your Mac — copy files to server
scp -r /Users/rameshvishwakarma/cricketVideo root@YOUR_SERVER_IP:/opt/cricket-api

# On server
cd /opt/cricket-api
docker compose up -d --build
```

---

## Step 5 — Check It's Running

```bash
# On server
docker compose ps             # should show "running"
docker compose logs -f        # live logs
curl http://localhost:8000/health    # should return {"status":"ok"}
```

Now test from your browser:
```
http://YOUR_SERVER_IP:8000
```

---

## Step 6 — Set Up a Domain + HTTPS (Optional but Recommended)

### Point your domain to the server
In your domain registrar (GoDaddy / Namecheap / Cloudflare):
- Add an **A record**: `cricket.yourdomain.com` → `YOUR_SERVER_IP`

### Install Certbot for free SSL

```bash
# On server
apt install certbot nginx -y

# Get SSL certificate
certbot certonly --standalone -d cricket.yourdomain.com \
  --email you@email.com --agree-tos --non-interactive

# Certs are now at:
# /etc/letsencrypt/live/cricket.yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/cricket.yourdomain.com/privkey.pem
```

### Update nginx.conf
```bash
# On server — edit nginx.conf
nano nginx.conf
```
Replace `your-domain.com` with `cricket.yourdomain.com`

```bash
# Update docker-compose.yml — update the Traefik label too
nano docker-compose.yml
```

```bash
# Restart with Nginx profile
docker compose --profile nginx up -d
```

Now visit: **https://cricket.yourdomain.com** ✅

---

## Step 7 — Keep Running After Reboots

```bash
# On server — already handled by restart: unless-stopped in docker-compose
# But also enable Docker to start on boot:
systemctl enable docker
```

---

## Step 8 — Firewall Setup

```bash
# On server
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw enable

# If running without Nginx (direct port 8000):
ufw allow 8000
```

---

## 🔄 Update the App After Code Changes

```bash
# On your Mac
git add . && git commit -m "update" && git push

# On server
cd /opt/cricket-api
git pull
docker compose up -d --build   # rebuilds and restarts
```

---

## 📊 Useful Server Commands

```bash
# View live logs
docker compose logs -f cricket-api

# Restart app
docker compose restart cricket-api

# Stop everything
docker compose down

# Check disk/memory
df -h      # disk
free -h    # memory
htop       # CPU + memory live

# Check uploaded videos
docker compose exec cricket-api ls -lh app/uploads/
```

---

## ☁️ Alternative: AWS EC2

If you prefer AWS:

```bash
# 1. Launch EC2 instance
#    - AMI: Ubuntu 22.04 LTS
#    - Type: t3.small (2GB RAM) or t3.medium (4GB RAM)
#    - Security Group: allow 22, 80, 443, 8000

# 2. SSH in
ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP

# 3. Same Docker setup as DigitalOcean (steps 3-8 above)
```

> [!NOTE]
> AWS t3.small = ~$15/mo. DigitalOcean $12/mo Droplet is simpler for POC.

---

## ☁️ Alternative: Hetzner VPS (Cheapest)

```bash
# CAX11 (ARM) = €3.79/mo — 2 vCPU, 4GB RAM, 40GB SSD
# Perfect for POC
# Same Ubuntu + Docker setup applies
```
Buy at: https://www.hetzner.com/cloud

---

## 📋 Deployment Checklist

```
[ ] Docker image builds successfully locally
[ ] docker compose up works locally at http://localhost:8000
[ ] Server provisioned (DigitalOcean / AWS / Hetzner)
[ ] Docker + Docker Compose installed on server
[ ] Code copied to server (git clone or scp)
[ ] docker compose up -d --build runs on server
[ ] http://SERVER_IP:8000/health returns {"status":"ok"}
[ ] Firewall rules set (ufw)
[ ] (Optional) Domain pointing to server IP
[ ] (Optional) SSL certificate via Certbot
[ ] (Optional) Nginx reverse proxy running
```

---

## 🔑 What Info You Need From Client

For a real deployment, ask client for:

| Item | Why |
|---|---|
| Server preference (DO / AWS / Hetzner) | Where to deploy |
| Domain name | For HTTPS setup |
| Expected concurrent users | Scaling decision |
| Max video size limit | Storage planning |
| Data retention policy | Uploaded video cleanup |

---

> [!TIP]
> **Cheapest production path for demo:**
> Hetzner CAX11 (€3.79/mo) + Docker + this guide = live in 20 minutes.

> [!IMPORTANT]
> MediaPipe needs at least **2 GB RAM**. Don't pick a 1 GB server — it will OOM-kill during processing.
