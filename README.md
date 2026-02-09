# 🤖 Telegram Subscription Management Bot

A professional, modular Telegram bot for managing subscription plans with a PostgreSQL backend, FastAPI dashboard integration, and robust subscription logic.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Telegram Bot API](https://img.shields.io/badge/Telegram%20Bot%20API-20.7-blue.svg)](https://python-telegram-bot.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-316192.svg)](https://www.postgresql.org/)
[![uv](https://img.shields.io/badge/uv-fast-purple)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

### Core Functionality
- 👤 **User Management** - Account creation with email validation
- 💳 **Smart Subscriptions** - Prorated upgrades/downgrades and auto-renewals
- 🌐 **Dashboard API** - Integrated FastAPI backend for the web dashboard
- 🔐 **Secure Auth** - Token-based authentication between Bot and Dashboard
- 📊 **Plan Details** - Detailed information for each subscription tier

### Admin Features
- 🔐 **Admin Panel** - Dedicated control panel for administrators
- 📈 **Statistics** - User and subscription analytics
- 💾 **Backup & Restore** - JSON-based database backups
- 👥 **User Management** - View and manage users
- 🔄 **Subscription Control** - Expire or cancel subscriptions

### Technical Features
- 🏗️ **Modular Architecture** - Separated concerns (handlers, database, api, payments)
- 🚀 **Unified Runner** - Single entry point (`start.py`) for Bot and API
- 🔄 **Auto Schema Init** - Automatic database initialization with fallback
- 📝 **Comprehensive Logging** - File rotation and proper error tracking
- 🔒 **Connection Pooling** - Efficient async database connections

## 📁 Project Structure


```
drelegrambot/
├── src/
│ ├── init.py # Package initialization
│ ├── start.py # 🚀 Unified entry point (Runs Bot + API)
│ ├── main.py # Bot polling logic
│ ├── api.py # FastAPI backend endpoints
│ ├── config.py # Configuration management
│ ├── database.py # Database CRUD operations
│ ├── handlers.py # Telegram command/callback handlers
│ ├── keyboards.py # Telegram keyboard layouts
│ ├── payment.py # Crypto gateway wrappers
│ ├── payment_manager.py # 🧠 Business logic (Proration/Calculations)
│ └── token_manager.py # 🔐 Secure token generation/hashing
├── backups/ # Database backups (auto-created)
├── logs/ # Application logs (auto-created)
├── utils.py # CLI management tool
├── schema.sql # Database schema
├── docker-compose.yaml # PostgreSQL container
├── requirements.txt # Python dependencies
└── README.md # This file             # This file
```
## 🚀 Quick Start

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Fast Python package installer)
- Docker and Docker Compose (for PostgreSQL)
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/telegram-subscription-bot.git
   cd telegram-subscription-bot
   ```

2. **Initialize environment with uv**
   ```bash
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Configure Environment**
   Set the required environment variables (see [Configuration](#-configuration) below). You can export them in your shell or use your preferred environment manager.

4. **Start PostgreSQL database**
   ```bash
   docker-compose up -d
   ```

5. **Verify database connection**
   ```bash
   uv run python utils.py check
   ```

6. **Run the bot**
   ```bash
   uv run python src/start.py
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

Use `uv run` to execute management commands safely:
```
# Check database connection
uv run python utils.py check

# View statistics
uv run python utils.py stats

# List recent users
uv run python utils.py users 20

# Create backup
uv run python utils.py backup

# List all backups
uv run python utils.py list-backups

# Restore from backup
uv run python utils.py restore backups/backup_20260107_120000.json

# Restore and clear existing data
uv run python utils.py restore backups/backup_20260107_120000.json --clear

# Manually expire old subscriptions
uv run python utils.py expire

# Show help
uv run python utils.py help
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
uv run python utils.py check

# View current stats
uv run python utils.py stats

# Create test backup
uv run python utils.py backup
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
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

## 📞 Support

For questions and support:
- 📧 Email: [support@example.com](mailto:support@example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/telegram-subscription-bot/issues)
- 💬 Telegram: [@yourusername](https://t.me/yourusername)

---

**Built with ❤️ using Python, PostgreSQL, and uv**
