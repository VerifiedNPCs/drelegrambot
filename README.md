# 🚀 Telegram Bot Deployment Guide

## Step-by-Step Instructions

### 📋 Prerequisites

1. **Server with Docker installed** (VPS, DigitalOcean, AWS EC2, etc.)
2. **Telegram Bot Token** from @BotFather

---

## 🎯 Part 1: Get Your Bot Token

1. Open Telegram and search for **@BotFather**
2. Send command: `/newbot`
3. Choose a name for your bot (e.g., "My Subscription Bot")
4. Choose a username (must end with 'bot', e.g., "mysubscription_bot")
5. Copy the **API token** you receive (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

---

## 🐳 Part 2: Deploy with Docker

### Method A: Deploy on VPS (DigitalOcean, AWS, etc.)

#### 1️⃣ Connect to Your Server
```bash
ssh root@your-server-ip
```

#### 2️⃣ Install Docker (if not installed)
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify installation
docker --version
docker-compose --version
```

#### 3️⃣ Create Project Directory
```bash
mkdir -p ~/subscription-bot
cd ~/subscription-bot
```

#### 4️⃣ Upload Your Files
Transfer these files to the server:
- main.py
- schema.sql
- Dockerfile
- docker-compose.yml
- requirements.txt

**Option A: Using SCP from your local machine:**
```bash
# From your local machine (not on server)
scp main.py schema.sql Dockerfile docker-compose.yml requirements.txt root@your-server-ip:~/subscription-bot/
```

**Option B: Using Git:**
```bash
# On server
git clone your-repo-url
cd your-repo-name
```

**Option C: Manual file creation:**
```bash
# Create each file manually using nano
nano main.py      # Paste content and save (Ctrl+X, Y, Enter)
nano schema.sql
nano Dockerfile
nano docker-compose.yml
nano requirements.txt
```

#### 5️⃣ Configure Environment Variables
```bash
# Edit .env file
nano .env
```

Add your bot token:
```bash
BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
DB_PASSWORD=your_strong_password_here
PORT=8080
```

Save: `Ctrl+X`, then `Y`, then `Enter`

#### 6️⃣ Start the Bot
```bash
# Build and start containers
docker-compose up -d

# Check if containers are running
docker-compose ps

# View bot logs
docker-compose logs -f bot
```

You should see:
```
subscription_bot | INFO | Starting Subscription Management Bot...
subscription_bot | INFO | Database pool created successfully
subscription_bot | INFO | Bot is running...
```

#### 7️⃣ Test Your Bot
1. Open Telegram
2. Search for your bot username
3. Send `/start` command
4. You should see the welcome message with buttons!

---

### Method B: Deploy Locally (for testing)

#### 1️⃣ Navigate to Project Directory
```bash
cd /path/to/subscription-bot
```

#### 2️⃣ Edit .env File
```bash
# Windows
notepad .env

# Mac/Linux
nano .env
```

Add your bot token:
```
BOT_TOKEN=your_token_here
DB_PASSWORD=postgres123
```

#### 3️⃣ Start with Docker Compose
```bash
docker-compose up -d
```

#### 4️⃣ Check Logs
```bash
docker-compose logs -f bot
```

---

## 🔧 Useful Commands

### Container Management
```bash
# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# Restart bot only
docker-compose restart bot

# View all logs
docker-compose logs -f

# View bot logs only
docker-compose logs -f bot

# View database logs
docker-compose logs -f postgres

# Check container status
docker-compose ps

# Rebuild after code changes
docker-compose up -d --build
```

### Database Management
```bash
# Access PostgreSQL shell
docker-compose exec postgres psql -U postgres -d subscription_bot

# Inside PostgreSQL shell:
# List all tables
\dt

# View users
SELECT * FROM users;

# View subscriptions
SELECT * FROM subscriptions;

# Exit
\q

# Backup database
docker-compose exec postgres pg_dump -U postgres subscription_bot > backup_$(date +%Y%m%d).sql

# Restore database
docker-compose exec -T postgres psql -U postgres subscription_bot < backup_20260106.sql
```

### Troubleshooting
```bash
# Check if containers are running
docker ps

# View all container logs
docker-compose logs

# Restart everything
docker-compose down && docker-compose up -d

# Remove everything and start fresh
docker-compose down -v
docker-compose up -d --build

# Check container resource usage
docker stats
```

---

## 🔐 Security Best Practices

### 1️⃣ Change Default Password
```bash
# In .env file, change:
DB_PASSWORD=use_a_very_strong_password_here_123!@#
```

### 2️⃣ Firewall Setup (Optional but Recommended)
```bash
# Allow SSH
ufw allow 22

# Enable firewall
ufw enable

# PostgreSQL port is only accessible inside Docker network
# No need to expose it externally
```

### 3️⃣ Keep Bot Token Secret
- Never commit .env to Git
- Never share your bot token
- Regenerate token if exposed (via @BotFather)

---

## 📊 Monitoring Your Bot

### Check if Bot is Running
```bash
# Method 1: Check container status
docker-compose ps

# Method 2: Check logs
docker-compose logs --tail=50 bot

# Method 3: Test in Telegram
# Send /start to your bot
```

### View Real-time Logs
```bash
# All logs
docker-compose logs -f

# Only bot logs
docker-compose logs -f bot

# Last 100 lines
docker-compose logs --tail=100 bot
```

---

## 🔄 Updating Your Bot

### Update Code
```bash
# 1. Edit main.py on your local machine
nano main.py

# 2. Rebuild and restart
docker-compose up -d --build

# 3. Check logs
docker-compose logs -f bot
```

### Update Database Schema
```bash
# 1. Edit schema.sql
nano schema.sql

# 2. Apply changes manually
docker-compose exec postgres psql -U postgres -d subscription_bot -f /docker-entrypoint-initdb.d/01-schema.sql
```

---

## 🛑 Stopping the Bot

### Temporary Stop
```bash
docker-compose stop
```

### Permanent Stop and Remove
```bash
# Stop and remove containers (keeps data)
docker-compose down

# Remove everything including data
docker-compose down -v
```

---

## 📱 Testing Your Bot

1. **Open Telegram**
2. **Search for your bot** by username
3. **Send `/start`** - You should see the welcome menu
4. **Test features:**
   - Click "Create Account" → Enter email
   - Click "Choose Plan" → Select a plan
   - Click "Subscription Status" → See days left
   - Click "Plan Details" → View plan information

---

## 🐛 Common Issues & Solutions

### Issue: "Bot not responding"
**Solution:**
```bash
# Check if container is running
docker-compose ps

# Check logs for errors
docker-compose logs bot

# Restart bot
docker-compose restart bot
```

### Issue: "Database connection failed"
**Solution:**
```bash
# Check if postgres is running
docker-compose ps postgres

# Check postgres logs
docker-compose logs postgres

# Restart postgres
docker-compose restart postgres
```

### Issue: "Invalid bot token"
**Solution:**
```bash
# 1. Verify token in .env file
cat .env

# 2. Get new token from @BotFather if needed
# 3. Update .env and restart
nano .env
docker-compose restart bot
```

### Issue: "Port already in use"
**Solution:**
```bash
# Find what's using port 5432
sudo lsof -i :5432

# Kill the process or change port in docker-compose.yml
```

---

## 💡 Production Tips

### 1️⃣ Use Environment Variables
Never hardcode secrets in code. Always use .env file.

### 2️⃣ Enable Logging
Logs are already configured in docker-compose.yml with rotation.

### 3️⃣ Regular Backups
```bash
# Add to crontab for daily backups
0 2 * * * cd /root/subscription-bot && docker-compose exec postgres pg_dump -U postgres subscription_bot > /backups/db_$(date +\%Y\%m\%d).sql
```

### 4️⃣ Monitor Resources
```bash
# Check memory/CPU usage
docker stats

# Set resource limits in docker-compose.yml if needed
```

### 5️⃣ Use Restart Policy
Already configured with `restart: always` in docker-compose.yml

---

## 🎉 Success Checklist

- [ ] Docker and Docker Compose installed
- [ ] Bot token obtained from @BotFather
- [ ] All files uploaded to server
- [ ] .env file configured with bot token
- [ ] Containers started with `docker-compose up -d`
- [ ] Bot responding to `/start` command in Telegram
- [ ] Database storing users and subscriptions
- [ ] Logs showing no errors

---

## 📞 Support

If you encounter issues:
1. Check logs: `docker-compose logs -f bot`
2. Verify containers: `docker-compose ps`
3. Test database: `docker-compose exec postgres psql -U postgres -d subscription_bot`

---

**Your bot should now be running! 🚀**

Test it by sending `/start` to your bot on Telegram.
