import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from database import DatabaseManager
from config import Config

# Global database manager
db_manager = DatabaseManager()


async def create_backup():
    """Create a database backup"""
    print("🔄 Creating backup...")
    await db_manager.initialize()
    backup_path = await db_manager.backup_to_json()
    if backup_path:
        print(f"✅ Backup created: {backup_path}")
    else:
        print("❌ Backup failed")
    await db_manager.close()


async def list_backups():
    """List all backups"""
    backups = await db_manager.list_backups()
    if not backups:
        print("📦 No backups found")
        return

    print(f"\n📦 Found {len(backups)} backup(s):\n")
    for i, backup in enumerate(backups, 1):
        print(f"  {i}. {backup['filename']}")
        print(f"     Size: {backup['size_mb']} MB")
        print(f"     Created: {backup['created']}")
        print()


async def restore_backup(backup_file: str, clear: bool = False):
    """Restore database from backup"""
    print(f"🔄 Restoring from {backup_file}...")
    await db_manager.initialize()

    if clear:
        confirm = input("⚠️  This will DELETE all existing data. Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Restore cancelled")
            await db_manager.close()
            return

    success = await db_manager.restore_from_json(backup_file, clear_existing=clear)
    if success:
        print("✅ Restore completed successfully")
    else:
        print("❌ Restore failed")
    await db_manager.close()


async def show_stats():
    """Show database statistics"""
    await db_manager.initialize()
    stats = await db_manager.get_statistics()

    print("\n📊 Database Statistics:\n")
    print(f"  👥 Total Users: {stats.get('total_users', 0)}")
    print(f"  ✅ Active Subscriptions: {stats.get('active_subscriptions', 0)}")
    print(f"  ❌ Expired Subscriptions: {stats.get('expired_subscriptions', 0)}")
    print(f"  🚫 Cancelled Subscriptions: {stats.get('cancelled_subscriptions', 0)}")

    if stats.get('subscriptions_by_plan'):
        print("\n  📦 Subscriptions by Plan:")
        for plan, count in stats['subscriptions_by_plan'].items():
            plan_info = Config.get_plan(plan)
            if plan_info:
                print(f"     {plan_info['emoji']} {plan_info['name']}: {count}")
    print()

    await db_manager.close()


async def check_db():
    """Check database connection and initialization"""
    print("🔍 Checking database connection...")
    success = await db_manager.initialize()
    if success:
        print("✅ Database connection successful")
        print("✅ Database initialized")

        # Show basic info
        stats = await db_manager.get_statistics()
        print(f"\n📊 Quick Stats:")
        print(f"   Users: {stats.get('total_users', 0)}")
        print(f"   Active Subs: {stats.get('active_subscriptions', 0)}")
    else:
        print("❌ Database connection failed")
    await db_manager.close()


async def list_users(limit: int = 10):
    """List recent users"""
    await db_manager.initialize()
    users = await db_manager.get_all_users(limit=limit)

    if not users:
        print("👥 No users found")
        await db_manager.close()
        return

    print(f"\n👥 Recent Users (showing {len(users)} of {await db_manager.count_users()}):\n")
    for i, user in enumerate(users, 1):
        print(f"  {i}. @{user['username']} ({user['user_id']})")
        print(f"     Email: {user['email']}")
        print(f"     Joined: {user['created_at'].strftime('%Y-%m-%d %H:%M')}")
        print()

    await db_manager.close()


async def expire_now():
    """Manually expire subscriptions"""
    print("🔄 Expiring old subscriptions...")
    await db_manager.initialize()
    count = await db_manager.expire_subscriptions()
    if count > 0:
        print(f"✅ Expired {count} subscription(s)")
    else:
        print("✓ No subscriptions to expire")
    await db_manager.close()


def print_help():
    """Print help message"""
    print("""
📚 Database Management Utility
================================

Usage: python utils.py <command> [args]

Commands:
  backup                    Create a database backup
  list-backups             List all available backups
  restore <file> [--clear] Restore from backup file
  stats                    Show database statistics
  check                    Check database connection
  users [limit]            List recent users (default: 10)
  expire                   Manually expire old subscriptions
  help                     Show this help message

Examples:
  python utils.py backup
  python utils.py list-backups
  python utils.py restore backups/backup_20260107_120000.json
  python utils.py restore backups/backup_20260107_120000.json --clear
  python utils.py stats
  python utils.py check
  python utils.py users 20
  python utils.py expire

Options:
  --clear    (restore) Clear existing data before restore

Note: Run this script from the project root directory
""")


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "backup":
            asyncio.run(create_backup())

        elif command == "list-backups":
            asyncio.run(list_backups())

        elif command == "restore":
            if len(sys.argv) < 3:
                print("❌ Error: Please specify backup file")
                print("   Usage: python utils.py restore <backup_file> [--clear]")
                sys.exit(1)
            backup_file = sys.argv[2]
            clear = "--clear" in sys.argv
            asyncio.run(restore_backup(backup_file, clear))

        elif command == "stats":
            asyncio.run(show_stats())

        elif command == "check":
            asyncio.run(check_db())

        elif command == "users":
            limit = 10
            if len(sys.argv) >= 3 and sys.argv[2].isdigit():
                limit = int(sys.argv[2])
            asyncio.run(list_users(limit))

        elif command == "expire":
            asyncio.run(expire_now())

        elif command == "help" or command == "-h" or command == "--help":
            print_help()

        else:
            print(f"❌ Unknown command: {command}")
            print("   Run 'python utils.py help' for available commands")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
