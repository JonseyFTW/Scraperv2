"""
Redis-based task queue for distributed scraping
Provides better job management than PostgreSQL row locking
"""
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

try:
    import redis
    from redis import Redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    Redis = None

from rich.console import Console
from rich.table import Table

console = Console()


class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 1
    NORMAL = 2
    LOW = 3
    RETRY = 4  # Failed tasks get lower priority


class TaskStatus(Enum):
    """Task status tracking"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Task:
    """Task data structure"""
    id: str
    type: str  # 'discover', 'csv_download', 'parse', 'image_scrape', 'image_download'
    data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.QUEUED
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    error: str = None
    worker_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if not self.id:
            self.id = str(uuid.uuid4())
            
    def to_json(self) -> str:
        """Serialize to JSON"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return json.dumps(data)
        
    @classmethod
    def from_json(cls, json_str: str) -> 'Task':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        data['priority'] = TaskPriority(data['priority'])
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


class RedisTaskQueue:
    """
    Redis-based task queue with priority, retries, and monitoring
    """
    
    def __init__(self, redis_url: str = None, worker_id: str = None):
        if not HAS_REDIS:
            raise ImportError("Redis not installed. Run: pip install redis")
            
        self.redis_url = redis_url or "redis://localhost:6379"
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.redis: Redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        
        # Queue keys
        self.queues = {
            TaskPriority.HIGH: "scraper:queue:high",
            TaskPriority.NORMAL: "scraper:queue:normal",
            TaskPriority.LOW: "scraper:queue:low",
            TaskPriority.RETRY: "scraper:queue:retry"
        }
        self.processing_key = "scraper:processing"
        self.completed_key = "scraper:completed"
        self.failed_key = "scraper:failed"
        self.stats_key = "scraper:stats"
        self.workers_key = "scraper:workers"
        
        # Test connection
        try:
            self.redis.ping()
            console.print(f"[green]Redis connected: {self.redis_url}[/green]")
            self.register_worker()
        except Exception as e:
            console.print(f"[red]Redis connection failed: {e}[/red]")
            raise
            
    def register_worker(self):
        """Register this worker"""
        self.redis.hset(self.workers_key, self.worker_id, json.dumps({
            "id": self.worker_id,
            "started": time.time(),
            "last_seen": time.time(),
            "tasks_completed": 0,
            "tasks_failed": 0
        }))
        
    def heartbeat(self):
        """Update worker heartbeat"""
        worker_data = self.redis.hget(self.workers_key, self.worker_id)
        if worker_data:
            data = json.loads(worker_data)
            data["last_seen"] = time.time()
            self.redis.hset(self.workers_key, self.worker_id, json.dumps(data))
            
    def push(self, task: Task) -> bool:
        """Push task to appropriate priority queue"""
        try:
            queue = self.queues[task.priority]
            self.redis.lpush(queue, task.to_json())
            self.redis.hincrby(self.stats_key, "total_queued", 1)
            self.redis.hincrby(self.stats_key, f"queued_{task.type}", 1)
            return True
        except Exception as e:
            console.print(f"[red]Failed to push task: {e}[/red]")
            return False
            
    def push_batch(self, tasks: List[Task]) -> int:
        """Push multiple tasks efficiently"""
        pipe = self.redis.pipeline()
        count = 0
        
        for task in tasks:
            queue = self.queues[task.priority]
            pipe.lpush(queue, task.to_json())
            count += 1
            
        pipe.hincrby(self.stats_key, "total_queued", count)
        pipe.execute()
        return count
        
    def pop(self, timeout: int = 0) -> Optional[Task]:
        """
        Pop highest priority task from queues
        Uses BRPOPLPUSH for atomic operation with timeout
        """
        # Try queues in priority order
        for priority in [TaskPriority.HIGH, TaskPriority.NORMAL, 
                        TaskPriority.LOW, TaskPriority.RETRY]:
            queue = self.queues[priority]
            
            # Atomic pop and push to processing queue
            task_json = self.redis.brpoplpush(
                queue, self.processing_key, timeout=timeout
            )
            
            if task_json:
                task = Task.from_json(task_json)
                task.status = TaskStatus.PROCESSING
                task.started_at = time.time()
                task.worker_id = self.worker_id
                
                # Update task in processing queue
                self.redis.lrem(self.processing_key, 1, task_json)
                self.redis.lpush(self.processing_key, task.to_json())
                
                # Update stats
                self.redis.hincrby(self.stats_key, "total_processing", 1)
                self.heartbeat()
                
                return task
                
        return None
        
    def complete(self, task: Task):
        """Mark task as completed"""
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        
        # Remove from processing
        self.redis.lrem(self.processing_key, 0, task.to_json())
        
        # Add to completed (with TTL)
        self.redis.setex(
            f"{self.completed_key}:{task.id}",
            3600,  # Keep for 1 hour
            task.to_json()
        )
        
        # Update stats
        self.redis.hincrby(self.stats_key, "total_completed", 1)
        self.redis.hincrby(self.stats_key, f"completed_{task.type}", 1)
        self.redis.hincrby(self.stats_key, "total_processing", -1)
        
        # Update worker stats
        worker_data = json.loads(self.redis.hget(self.workers_key, self.worker_id))
        worker_data["tasks_completed"] += 1
        self.redis.hset(self.workers_key, self.worker_id, json.dumps(worker_data))
        
    def fail(self, task: Task, error: str):
        """Mark task as failed and potentially retry"""
        task.error = error
        task.retry_count += 1
        
        # Remove from processing
        self.redis.lrem(self.processing_key, 0, task.to_json())
        
        if task.retry_count < task.max_retries:
            # Retry with lower priority
            task.priority = TaskPriority.RETRY
            task.status = TaskStatus.QUEUED
            self.push(task)
            console.print(f"[yellow]Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries})[/yellow]")
        else:
            # Max retries exceeded
            task.status = TaskStatus.FAILED
            self.redis.setex(
                f"{self.failed_key}:{task.id}",
                86400,  # Keep for 24 hours
                task.to_json()
            )
            self.redis.hincrby(self.stats_key, "total_failed", 1)
            console.print(f"[red]Task {task.id} failed permanently: {error}[/red]")
            
        # Update stats
        self.redis.hincrby(self.stats_key, "total_processing", -1)
        
        # Update worker stats
        worker_data = json.loads(self.redis.hget(self.workers_key, self.worker_id))
        worker_data["tasks_failed"] += 1
        self.redis.hset(self.workers_key, self.worker_id, json.dumps(worker_data))
        
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        stats = {}
        
        # Queue sizes
        for priority, queue in self.queues.items():
            stats[f"queue_{priority.name.lower()}"] = self.redis.llen(queue)
            
        # Processing queue
        stats["processing"] = self.redis.llen(self.processing_key)
        
        # Overall stats
        overall = self.redis.hgetall(self.stats_key)
        stats.update({k: int(v) for k, v in overall.items()})
        
        # Active workers
        workers = self.redis.hgetall(self.workers_key)
        active_workers = []
        stale_threshold = time.time() - 60  # Consider stale after 60 seconds
        
        for worker_json in workers.values():
            worker = json.loads(worker_json)
            if worker["last_seen"] > stale_threshold:
                active_workers.append(worker)
                
        stats["active_workers"] = len(active_workers)
        stats["workers"] = active_workers
        
        return stats
        
    def clear_stale_tasks(self, timeout: int = 300):
        """
        Clear tasks that have been processing too long
        Default: 5 minutes
        """
        processing = self.redis.lrange(self.processing_key, 0, -1)
        stale_threshold = time.time() - timeout
        cleared = 0
        
        for task_json in processing:
            task = Task.from_json(task_json)
            if task.started_at and task.started_at < stale_threshold:
                # Task is stale, move back to queue
                self.redis.lrem(self.processing_key, 1, task_json)
                task.status = TaskStatus.QUEUED
                task.priority = TaskPriority.RETRY
                self.push(task)
                cleared += 1
                console.print(f"[yellow]Cleared stale task {task.id}[/yellow]")
                
        return cleared
        
    def flush_all(self):
        """Clear all queues (dangerous!)"""
        for queue in self.queues.values():
            self.redis.delete(queue)
        self.redis.delete(self.processing_key)
        self.redis.delete(self.stats_key)
        console.print("[yellow]All queues flushed[/yellow]")


def show_queue_stats(redis_url: str = None):
    """Display queue statistics"""
    queue = RedisTaskQueue(redis_url)
    stats = queue.get_stats()
    
    # Queue table
    table = Table(title="Task Queue Status", show_header=True, header_style="bold cyan")
    table.add_column("Queue", style="white", min_width=15)
    table.add_column("Count", justify="right", style="green", min_width=10)
    
    table.add_row("High Priority", str(stats.get("queue_high", 0)))
    table.add_row("Normal Priority", str(stats.get("queue_normal", 0)))
    table.add_row("Low Priority", str(stats.get("queue_low", 0)))
    table.add_row("Retry Queue", str(stats.get("queue_retry", 0)))
    table.add_row("Processing", str(stats.get("processing", 0)))
    
    table.add_section()
    table.add_row("Total Queued", str(stats.get("total_queued", 0)))
    table.add_row("Total Completed", str(stats.get("total_completed", 0)))
    table.add_row("Total Failed", str(stats.get("total_failed", 0)))
    
    table.add_section()
    table.add_row("Active Workers", str(stats.get("active_workers", 0)))
    
    console.print(table)
    
    # Worker details
    if stats.get("workers"):
        worker_table = Table(title="Active Workers", show_header=True, header_style="bold magenta")
        worker_table.add_column("Worker ID", style="white")
        worker_table.add_column("Uptime", justify="right", style="yellow")
        worker_table.add_column("Completed", justify="right", style="green")
        worker_table.add_column("Failed", justify="right", style="red")
        
        for worker in stats["workers"]:
            uptime = int(time.time() - worker["started"])
            uptime_str = f"{uptime // 60}m {uptime % 60}s"
            worker_table.add_row(
                worker["id"],
                uptime_str,
                str(worker["tasks_completed"]),
                str(worker["tasks_failed"])
            )
            
        console.print(worker_table)


if __name__ == "__main__":
    # Test/demo
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        show_queue_stats()
    else:
        console.print("Testing Redis task queue...")
        
        queue = RedisTaskQueue()
        
        # Push some test tasks
        for i in range(5):
            task = Task(
                id=f"test_{i}",
                type="test",
                data={"index": i},
                priority=TaskPriority.NORMAL if i % 2 else TaskPriority.HIGH
            )
            queue.push(task)
            
        console.print("[green]Pushed 5 test tasks[/green]")
        
        # Pop and complete tasks
        while True:
            task = queue.pop(timeout=1)
            if not task:
                break
                
            console.print(f"Processing task {task.id}...")
            queue.complete(task)
            
        show_queue_stats()