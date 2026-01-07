# 🤖 Telegram Subscription Management Bot

A professional, modular Telegram bot for managing subscription plans with PostgreSQL backend, admin panel, and comprehensive CRUD operations.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Telegram Bot API](https://img.shields.io/badge/Telegram%20Bot%20API-20.7-blue.svg)](https://python-telegram-bot.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-316192.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

### Core Functionality
- 👤 **User Management** - Account creation with email validation
- 💳 **Subscription Plans** - Multiple tiers (Standard, Pro, Business+, Enterprise+)
- ⏰ **Status Tracking** - Real-time subscription status and days remaining
- 📊 **Plan Details** - Detailed information for each subscription tier
- 🎨 **Modern UI** - Glass-style button interface

### Admin Features
- 🔐 **Admin Panel** - Dedicated control panel for administrators
- 📈 **Statistics** - User and subscription analytics
- 💾 **Backup & Restore** - JSON-based database backups
- 👥 **User Management** - View and manage users
- 🔄 **Subscription Control** - Expire or cancel subscriptions

### Technical Features
- 🏗️ **Modular Architecture** - Separated concerns (handlers, keyboards, database)
- 🔄 **Auto Schema Init** - Automatic database initialization with fallback
- 📝 **Comprehensive Logging** - File rotation and proper error tracking
- ⚙️ **Background Jobs** - Auto-expiration and scheduled backups
- 🛠️ **CLI Tools** - Command-line utilities for management
- 🔒 **Connection Pooling** - Efficient async database connections

## 📁 Project Structure

```
drelegrambot/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration management
│   ├── database.py          # Database CRUD operations
│   ├── keyboards.py         # Telegram keyboard layouts
│   ├── handlers.py          # Command and callback handlers
│   └── main.py              # Application entry point
├── backups/                 # Database backups (auto-created)
├── logs/                    # Application logs (auto-created)
├── utils.py                 # CLI management tool
├── schema.sql               # Database schema
├── docker-compose.yaml      # PostgreSQL container
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- Your Telegram User ID (from [@userinfobot](https://t.me/userinfobot))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/telegram-subscription-bot.git
   cd telegram-subscription-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   
   Create `.env` file in project root:
   ```env
   # Bot Configuration
   BOT_TOKEN=your_telegram_bot_token_here
   
   # Database Configuration
   DATABASE_URL=postgresql://userdrelegram:yourpassword@localhost:9204/drelegrambotdb
   DB_MIN_POOL_SIZE=2
   DB_MAX_POOL_SIZE=10
   DB_COMMAND_TIMEOUT=60
   
   # Admin Configuration (comma-separated Telegram user IDs)
   ADMIN_USER_IDS=123456789,987654321
   
   # Backup Configuration
   BACKUP_DIR=./backups
   AUTO_BACKUP_ENABLED=true
   AUTO_BACKUP_INTERVAL_HOURS=24
   
   # Logging Configuration
   LOG_LEVEL=INFO
   LOG_DIR=./logs
   ```

4. **Start PostgreSQL database**
   ```bash
   docker-compose up -d
   ```

5. **Verify database connection**
   ```bash
   python utils.py check
   ```

6. **Run the bot**
   ```bash
   cd src
   python main.py
   ```

## 💳 Subscription Plans

| Plan | Price | Duration | Features |
|------|-------|----------|----------|
| 📦 Standard | $9.99/month | 30 days | Basic features, 5 API calls/day, Email support |
| ⭐ Pro | $29.99/month | 30 days | All Standard + 100 API calls/day, Priority support, Analytics |
| 💼 Business+ | $79.99/month | 30 days | All Pro + Unlimited API, 24/7 support, Custom integrations |
| 🏢 Enterprise+ | Custom | 30 days | All Business+ + Unlimited accounts, SLA, On-premise |

*Plans can be customized in `src/config.py`*

## 🎮 Bot Commands

### User Commands
- `/start` - Show main menu and welcome message
- `/account` - View account information
- `/plans` - Browse available subscription plans
- `/status` - Check current subscription status
- `/help` - Display help information

### Admin Commands
- `/admin` - Open admin control panel
- `/stats` - View bot statistics
- `/backup` - Create database backup manually

## 🛠️ CLI Management Tools

Run from project root:

```bash
# Check database connection
python utils.py check

# View statistics
python utils.py stats

# List recent users
python utils.py users 20

# Create backup
python utils.py backup

# List all backups
python utils.py list-backups

# Restore from backup
python utils.py restore backups/backup_20260107_120000.json

# Restore and clear existing data
python utils.py restore backups/backup_20260107_120000.json --clear

# Manually expire old subscriptions
python utils.py expire

# Show help
python utils.py help
```

## 🗄️ Database Schema

### Users Table
```sql
user_id BIGINT PRIMARY KEY
username VARCHAR(255)
email VARCHAR(255) UNIQUE
created_at TIMESTAMP
updated_at TIMESTAMP
```

### Subscriptions Table
```sql
id SERIAL PRIMARY KEY
user_id BIGINT (FK to users)
plan VARCHAR(50) (standard|pro|business+|enterprise+)
status VARCHAR(20) (active|expired|cancelled)
start_date TIMESTAMP
end_date TIMESTAMP
created_at TIMESTAMP
updated_at TIMESTAMP
```

## 📊 Features Breakdown

### User Management
- Create account with email validation
- View account information
- Update user details (username, email)
- Delete user (cascades to subscriptions)
- List all users with pagination

### Subscription Management
- Create new subscription (auto-cancels existing active ones)
- Get active subscription for user
- View subscription history
- Update subscription status
- Cancel subscription
- Delete subscription
- Check days remaining
- Find expiring subscriptions

### Statistics & Analytics
- Total users count
- Active subscriptions count
- Expired subscriptions count
- Cancelled subscriptions count
- Subscriptions by plan breakdown

### Backup & Restore
- Export to JSON format
- Import from JSON
- List all backups with metadata
- Automatic scheduled backups
- Manual backup creation

## 🔧 Configuration

All configuration is managed through environment variables and `src/config.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `BOT_TOKEN` | Telegram Bot API token | Required |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `ADMIN_USER_IDS` | Comma-separated admin Telegram IDs | Empty |
| `DB_MIN_POOL_SIZE` | Minimum database connections | 2 |
| `DB_MAX_POOL_SIZE` | Maximum database connections | 10 |
| `AUTO_BACKUP_ENABLED` | Enable automatic backups | false |
| `AUTO_BACKUP_INTERVAL_HOURS` | Backup interval in hours | 24 |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING) | INFO |

## 🏗️ Architecture

### Modular Design
- **config.py** - Centralized configuration with validation
- **database.py** - Complete database layer with CRUD operations
- **keyboards.py** - Reusable Telegram keyboard layouts
- **handlers.py** - Command and callback handlers
- **main.py** - Application entry point and lifecycle management

### Key Patterns
- **Dependency Injection** - Database manager injected into handlers
- **Separation of Concerns** - Each module has single responsibility
- **Error Handling** - Comprehensive try-catch blocks with logging
- **Connection Pooling** - Async PostgreSQL connection pool
- **Background Jobs** - Scheduled tasks for maintenance

## 📝 Development

### Adding New Features

1. **Add database operations** in `src/database.py`
2. **Add configuration** in `src/config.py`
3. **Add keyboard layouts** in `src/keyboards.py`
4. **Add handlers** in `src/handlers.py`
5. **Register handlers** in `src/main.py`

### Testing

```bash
# Test database connection
python utils.py check

# View current stats
python utils.py stats

# Create test backup
python utils.py backup
```

## 🐛 Troubleshooting

### Database Connection Issues
```bash
# Check if database is running
docker-compose ps

# View database logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### Bot Not Starting
```bash
# Check configuration
python utils.py check

# View bot logs
tail -f logs/bot.log

# Verify bot token
curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe
```

### Import Errors
```bash
# Ensure you're in the correct directory
cd src
python main.py

# Or use module syntax
python -m src.main
```

## 📈 Production Deployment

### Security Checklist
- [ ] Use strong database passwords
- [ ] Store sensitive data in environment variables
- [ ] Enable SSL/TLS for database connections
- [ ] Set up admin user IDs
- [ ] Enable automatic backups
- [ ] Configure log rotation

### Monitoring
- Set up log aggregation (ELK, Papertrail)
- Monitor database performance
- Set up alerts for errors
- Regular backup verification

### Scaling
- Increase database pool size if needed
- Deploy multiple bot instances
- Use Redis for caching (optional)
- Implement rate limiting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [python-telegram-bot](https://python-telegram-bot.org/) - Telegram Bot API wrapper
- [asyncpg](https://github.com/MagicStack/asyncpg) - Fast PostgreSQL client
- [python-dotenv](https://github.com/theskumar/python-dotenv) - Environment management

## 📞 Support

For questions and support:
- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/telegram-subscription-bot/issues)
- 💬 Telegram: [@yourusername](https://t.me/yourusername)

---

**Built with ❤️ using Python, PostgreSQL, and python-telegram-bot**
