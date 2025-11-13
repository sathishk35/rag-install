#!/usr/bin/env python3
"""
User Creation and Management Script for RAG System
Creates default users and manages user permissions
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add RAG system to path
sys.path.append('/opt/rag-system')

from core.security_manager import SecurityManager

class UserManager:
    def __init__(self):
        self.config = self._load_config()
        self.security_manager = SecurityManager(self.config)
        
    def _load_config(self):
        """Load configuration"""
        config_path = Path('/opt/rag-system/config/rag_config.yaml')
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "rag_metadata",
                    "user": "rag-system"
                }
            }
    
    async def create_default_users(self):
        """Create default system users"""
        print("Creating default users for RAG system...")
        
        users = [
            {
                'user_id': 'admin',
                'security_clearance': 'classified',
                'domains': ['drivers', 'embedded', 'radar', 'rf', 'ew', 'ate', 'general'],
                'description': 'System administrator with full access'
            },
            {
                'user_id': 'senior_engineer', 
                'security_clearance': 'restricted',
                'domains': ['drivers', 'embedded', 'radar', 'ate', 'general'],
                'description': 'Senior engineer with restricted access'
            },
            {
                'user_id': 'engineer',
                'security_clearance': 'confidential',
                'domains': ['drivers', 'embedded', 'general'],
                'description': 'Engineer with confidential access'
            },
            {
                'user_id': 'developer',
                'security_clearance': 'internal',
                'domains': ['drivers', 'general'],
                'description': 'Developer with internal access'
            },
            {
                'user_id': 'guest',
                'security_clearance': 'public',
                'domains': ['general'],
                'description': 'Guest user with public access only'
            }
        ]
        
        created_users = []
        failed_users = []
        
        for user in users:
            try:
                success = await self.security_manager.set_user_clearance(
                    user_id=user['user_id'],
                    security_level=user['security_clearance'],
                    domains=user['domains'],
                    admin_user='system'
                )
                
                if success:
                    print(f"✓ Created user: {user['user_id']} ({user['security_clearance']})")
                    created_users.append(user['user_id'])
                else:
                    print(f"✗ Failed to create user: {user['user_id']}")
                    failed_users.append(user['user_id'])
                    
            except Exception as e:
                print(f"✗ Error creating user {user['user_id']}: {e}")
                failed_users.append(user['user_id'])
        
        print(f"\nSummary:")
        print(f"Created: {len(created_users)} users")
        print(f"Failed: {len(failed_users)} users")
        
        if created_users:
            print(f"Successfully created: {', '.join(created_users)}")
        if failed_users:
            print(f"Failed to create: {', '.join(failed_users)}")
        
        return created_users, failed_users
    
    async def create_user(self, user_id: str, security_level: str, domains: list, admin_user: str = 'admin'):
        """Create a single user"""
        try:
            success = await self.security_manager.set_user_clearance(
                user_id=user_id,
                security_level=security_level,
                domains=domains,
                admin_user=admin_user
            )
            
            if success:
                print(f"✓ Successfully created user: {user_id}")
                print(f"  Security Level: {security_level}")
                print(f"  Domains: {', '.join(domains)}")
                return True
            else:
                print(f"✗ Failed to create user: {user_id}")
                return False
                
        except Exception as e:
            print(f"✗ Error creating user {user_id}: {e}")
            return False
    
    async def list_users(self):
        """List all users in the system"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"],
                database=self.config["database"]["database"],
                user=self.config["database"]["user"]
            )
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, security_clearance, domains, created_at, updated_at
                FROM user_access 
                ORDER BY security_clearance DESC, user_id
            """)
            
            results = cursor.fetchall()
            
            if not results:
                print("No users found in the system.")
                return
            
            print(f"{'User ID':<20} {'Security Level':<15} {'Domains':<30} {'Created':<20}")
            print("-" * 85)
            
            for row in results:
                user_id, security_level, domains, created_at, updated_at = row
                domains_str = ', '.join(domains) if domains else 'None'
                created_str = created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'Unknown'
                
                print(f"{user_id:<20} {security_level:<15} {domains_str:<30} {created_str:<20}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error listing users: {e}")
    
    async def update_user(self, user_id: str, security_level: str = None, domains: list = None, admin_user: str = 'admin'):
        """Update an existing user"""
        try:
            # Get current user info
            current_info = await self.security_manager.get_user_clearance(user_id)
            
            if not current_info or current_info.get('user_id') != user_id:
                print(f"User {user_id} not found")
                return False
            
            # Use current values if not provided
            new_security_level = security_level or current_info['security_clearance']
            new_domains = domains or current_info['domains']
            
            success = await self.security_manager.set_user_clearance(
                user_id=user_id,
                security_level=new_security_level,
                domains=new_domains,
                admin_user=admin_user
            )
            
            if success:
                print(f"✓ Successfully updated user: {user_id}")
                print(f"  Security Level: {current_info['security_clearance']} → {new_security_level}")
                print(f"  Domains: {', '.join(current_info['domains'])} → {', '.join(new_domains)}")
                return True
            else:
                print(f"✗ Failed to update user: {user_id}")
                return False
                
        except Exception as e:
            print(f"✗ Error updating user {user_id}: {e}")
            return False
    
    async def delete_user(self, user_id: str, admin_user: str = 'admin'):
        """Delete a user (soft delete by setting to inactive)"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.config["database"]["host"],
                port=self.config["database"]["port"], 
                database=self.config["database"]["database"],
                user=self.config["database"]["user"]
            )
            
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute("SELECT user_id FROM user_access WHERE user_id = %s", (user_id,))
            if not cursor.fetchone():
                print(f"User {user_id} not found")
                return False
            
            # Delete user
            cursor.execute("DELETE FROM user_access WHERE user_id = %s", (user_id,))
            
            # Invalidate user sessions
            cursor.execute("UPDATE user_sessions SET active = false WHERE user_id = %s", (user_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"✓ Successfully deleted user: {user_id}")
            return True
            
        except Exception as e:
            print(f"✗ Error deleting user {user_id}: {e}")
            return False
    
    async def get_user_info(self, user_id: str):
        """Get detailed information about a user"""
        try:
            user_info = await self.security_manager.get_user_clearance(user_id)
            
            if not user_info or user_info.get('user_id') != user_id:
                print(f"User {user_id} not found")
                return
            
            print(f"User Information: {user_id}")
            print("-" * 40)
            print(f"Security Clearance: {user_info['security_clearance']}")
            print(f"Domains: {', '.join(user_info['domains'])}")
            print(f"Created: {user_info['created_at']}")
            print(f"Updated: {user_info['updated_at']}")
            
            # Get user activity summary
            try:
                activity = await self.security_manager.get_user_activity_summary(user_id, days=7)
                print(f"\nRecent Activity (7 days):")
                print(f"Queries: {activity.get('query_count', 0)}")
                print(f"Active Days: {activity.get('active_days', 0)}")
                print(f"Violations: {len(activity.get('violations', []))}")
                print(f"Risk Score: {activity.get('risk_score', 0):.2f}")
                
            except Exception:
                print("Could not retrieve activity information")
            
        except Exception as e:
            print(f"Error getting user info: {e}")
    
    async def bulk_create_users(self, users_file: str, admin_user: str = 'admin'):
        """Create users from JSON file"""
        try:
            with open(users_file, 'r') as f:
                users_data = json.load(f)
            
            if not isinstance(users_data, list):
                print("Error: Users file should contain a list of user objects")
                return False
            
            created_count = 0
            failed_count = 0
            
            for user_data in users_data:
                required_fields = ['user_id', 'security_clearance', 'domains']
                if not all(field in user_data for field in required_fields):
                    print(f"✗ Skipping invalid user data: {user_data}")
                    failed_count += 1
                    continue
                
                success = await self.create_user(
                    user_id=user_data['user_id'],
                    security_level=user_data['security_clearance'],
                    domains=user_data['domains'],
                    admin_user=admin_user
                )
                
                if success:
                    created_count += 1
                else:
                    failed_count += 1
            
            print(f"\nBulk creation summary:")
            print(f"Created: {created_count} users")
            print(f"Failed: {failed_count} users")
            
            return created_count > 0
            
        except Exception as e:
            print(f"Error in bulk user creation: {e}")
            return False

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RAG System User Management')
    parser.add_argument('action', choices=[
        'create-defaults', 'create-user', 'list-users', 'update-user', 
        'delete-user', 'get-user-info', 'bulk-create'
    ], help='Action to perform')
    
    # User creation arguments
    parser.add_argument('--user-id', help='User ID')
    parser.add_argument('--security-level', choices=[
        'public', 'internal', 'confidential', 'restricted', 'classified'
    ], help='Security clearance level')
    parser.add_argument('--domains', nargs='+', choices=[
        'drivers', 'embedded', 'radar', 'rf', 'ew', 'ate', 'general'
    ], help='Accessible domains')
    parser.add_argument('--admin-user', default='admin', help='Admin user performing the action')
    parser.add_argument('--users-file', help='JSON file containing users data for bulk creation')
    
    args = parser.parse_args()
    
    # Validate required arguments for different actions
    if args.action == 'create-user':
        if not all([args.user_id, args.security_level, args.domains]):
            print("Error: --user-id, --security-level, and --domains are required for create-user")
            return 1
    
    elif args.action in ['update-user', 'delete-user', 'get-user-info']:
        if not args.user_id:
            print(f"Error: --user-id is required for {args.action}")
            return 1
    
    elif args.action == 'bulk-create':
        if not args.users_file:
            print("Error: --users-file is required for bulk-create")
            return 1
    
    # Create user manager and execute action
    user_manager = UserManager()
    
    try:
        if args.action == 'create-defaults':
            await user_manager.create_default_users()
            
        elif args.action == 'create-user':
            await user_manager.create_user(
                user_id=args.user_id,
                security_level=args.security_level,
                domains=args.domains,
                admin_user=args.admin_user
            )
            
        elif args.action == 'list-users':
            await user_manager.list_users()
            
        elif args.action == 'update-user':
            await user_manager.update_user(
                user_id=args.user_id,
                security_level=args.security_level,
                domains=args.domains,
                admin_user=args.admin_user
            )
            
        elif args.action == 'delete-user':
            await user_manager.delete_user(
                user_id=args.user_id,
                admin_user=args.admin_user
            )
            
        elif args.action == 'get-user-info':
            await user_manager.get_user_info(args.user_id)
            
        elif args.action == 'bulk-create':
            await user_manager.bulk_create_users(
                users_file=args.users_file,
                admin_user=args.admin_user
            )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        user_manager.security_manager.close()

if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))