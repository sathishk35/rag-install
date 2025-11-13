"""
Database Manager with Connection Pooling
Provides efficient database operations with connection pooling and async support
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager, contextmanager
import psycopg2
from psycopg2 import pool, extras
from dataclasses import dataclass
import time

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    user: str
    password: Optional[str] = None
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    idle_timeout: int = 3600

class DatabaseManager:
    """
    Database manager with connection pooling
    Provides efficient database operations with proper resource management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract database configuration
        db_config = config.get('database', {})
        self.db_config = DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'rag_metadata'),
            user=db_config.get('user', 'rag-system'),
            password=db_config.get('password'),
            min_connections=db_config.get('pool_size', 5),
            max_connections=db_config.get('max_overflow', 20) + db_config.get('pool_size', 5),
            connection_timeout=db_config.get('pool_timeout', 30),
        )

        # Initialize connection pool
        self.connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self._init_connection_pool()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_query_time': 0.0,
            'pool_gets': 0,
            'pool_puts': 0,
        }

    def _init_connection_pool(self):
        """Initialize the connection pool"""
        try:
            # Build connection parameters
            conn_params = {
                'host': self.db_config.host,
                'port': self.db_config.port,
                'database': self.db_config.database,
                'user': self.db_config.user,
            }

            # Add password if available (from env or config)
            import os
            password = os.environ.get('POSTGRES_PASSWORD') or self.db_config.password
            if password:
                conn_params['password'] = password

            # Create connection pool
            self.connection_pool = pool.ThreadedConnectionPool(
                minconn=self.db_config.min_connections,
                maxconn=self.db_config.max_connections,
                **conn_params
            )

            self.logger.info(
                f"Database connection pool initialized: "
                f"min={self.db_config.min_connections}, "
                f"max={self.db_config.max_connections}"
            )

            # Test connection
            conn = self.get_connection()
            conn.close()
            self.put_connection(conn)

        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool

        Usage:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self.connection_pool.getconn()
            self.stats['pool_gets'] += 1
            yield conn
        except Exception as e:
            self.logger.error(f"Error getting connection: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
                self.stats['pool_puts'] += 1

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Context manager for getting a cursor

        Usage:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT ...")
                results = cursor.fetchall()
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error in cursor operation: {e}")
                raise
            finally:
                cursor.close()

    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch: bool = True) -> Optional[List[Tuple]]:
        """
        Execute a query and return results

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results (False for INSERT/UPDATE/DELETE)

        Returns:
            List of result tuples if fetch=True, else None
        """
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                self.stats['total_queries'] += 1
                self.stats['successful_queries'] += 1

                if fetch:
                    results = cursor.fetchall()
                    return results
                return None

        except Exception as e:
            self.stats['failed_queries'] += 1
            self.logger.error(f"Query execution error: {e}")
            self.logger.error(f"Query: {query}")
            raise
        finally:
            elapsed = time.time() - start_time
            self.stats['total_query_time'] += elapsed

    def execute_query_dict(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return results as list of dictionaries

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        start_time = time.time()
        try:
            with self.get_cursor(cursor_factory=extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                self.stats['total_queries'] += 1
                self.stats['successful_queries'] += 1
                results = cursor.fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            self.stats['failed_queries'] += 1
            self.logger.error(f"Query execution error: {e}")
            raise
        finally:
            elapsed = time.time() - start_time
            self.stats['total_query_time'] += elapsed

    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute a query multiple times with different parameters

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        start_time = time.time()
        try:
            with self.get_cursor() as cursor:
                cursor.executemany(query, params_list)
                self.stats['total_queries'] += len(params_list)
                self.stats['successful_queries'] += len(params_list)
                return cursor.rowcount

        except Exception as e:
            self.stats['failed_queries'] += len(params_list)
            self.logger.error(f"Batch execution error: {e}")
            raise
        finally:
            elapsed = time.time() - start_time
            self.stats['total_query_time'] += elapsed

    def execute_batch(self, query: str, params_list: List[Tuple], page_size: int = 100) -> int:
        """
        Execute a batch insert/update operation efficiently

        Args:
            query: SQL query string
            params_list: List of parameter tuples
            page_size: Number of records per batch

        Returns:
            Total number of rows affected
        """
        start_time = time.time()
        total_rows = 0

        try:
            with self.get_cursor() as cursor:
                extras.execute_batch(cursor, query, params_list, page_size=page_size)
                total_rows = cursor.rowcount
                self.stats['total_queries'] += 1
                self.stats['successful_queries'] += 1
                return total_rows

        except Exception as e:
            self.stats['failed_queries'] += 1
            self.logger.error(f"Batch execution error: {e}")
            raise
        finally:
            elapsed = time.time() - start_time
            self.stats['total_query_time'] += elapsed

    def insert_returning(self, query: str, params: Optional[Tuple] = None) -> Any:
        """
        Execute INSERT query and return the inserted ID

        Args:
            query: INSERT query with RETURNING clause
            params: Query parameters

        Returns:
            The returned value (usually ID)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return result[0] if result else None

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            )
        """
        result = self.execute_query(query, (table_name,))
        return result[0][0] if result else False

    def create_table_if_not_exists(self, create_query: str):
        """Create table if it doesn't exist"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute(create_query)
                self.logger.info("Table created or already exists")
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'failed_queries': self.stats['failed_queries'],
            'avg_query_time': (
                self.stats['total_query_time'] / self.stats['total_queries']
                if self.stats['total_queries'] > 0 else 0
            ),
            'pool_gets': self.stats['pool_gets'],
            'pool_puts': self.stats['pool_puts'],
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = time.time()
            result = self.execute_query("SELECT 1")
            latency = time.time() - start_time

            return {
                'status': 'healthy',
                'latency_ms': round(latency * 1000, 2),
                'pool_stats': self.get_pool_stats()
            }
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    def close_all_connections(self):
        """Close all connections in the pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("All database connections closed")

    def __del__(self):
        """Cleanup on deletion"""
        self.close_all_connections()


class AsyncDatabaseManager:
    """
    Async database manager using asyncpg for true async operations
    Note: Requires asyncpg to be installed
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pool = None

        # Extract database configuration
        db_config = config.get('database', {})
        self.db_config = DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'rag_metadata'),
            user=db_config.get('user', 'rag-system'),
            password=db_config.get('password'),
            min_connections=db_config.get('pool_size', 5),
            max_connections=db_config.get('max_overflow', 20) + db_config.get('pool_size', 5),
        )

    async def init_pool(self):
        """Initialize async connection pool"""
        try:
            import asyncpg

            # Build connection parameters
            import os
            password = os.environ.get('POSTGRES_PASSWORD') or self.db_config.password

            self.pool = await asyncpg.create_pool(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.user,
                password=password,
                min_size=self.db_config.min_connections,
                max_size=self.db_config.max_connections,
            )

            self.logger.info("Async database pool initialized")

        except ImportError:
            self.logger.warning(
                "asyncpg not installed. Using sync database manager. "
                "Install with: pip install asyncpg"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize async pool: {e}")
            raise

    async def execute(self, query: str, *args) -> str:
        """Execute a query without returning results"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return all results"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a query and return single row"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and return single value"""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute_many(self, query: str, args_list: List[Tuple]) -> None:
        """Execute query multiple times"""
        async with self.pool.acquire() as conn:
            await conn.executemany(query, args_list)

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self.logger.info("Async database pool closed")
