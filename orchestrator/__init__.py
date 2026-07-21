"""
Multi-Agent Orchestration System.

Features:
- Agent creation and management
- Task decomposition
- Parallel and sequential execution
- Agent communication (message passing)
- Shared memory/state
- Workflow definitions
- Error recovery
- Result aggregation

Agent Types:
- Worker: Executes specific tasks
- Coordinator: Manages other agents
- Specialist: Domain-specific expertise
- Monitor: Watches and reports
- Router: Directs tasks to appropriate agents
"""

import os
import json
import time
import asyncio
import inspect
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, AsyncIterator
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod


class AgentRole(Enum):
    """Agent roles in the system."""
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    MONITOR = "monitor"
    ROUTER = "router"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Agent status."""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentMessage:
    """Message between agents."""
    id: str
    sender_id: str
    receiver_id: str
    content: Any
    message_type: str = "data"
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class Task:
    """Task to be executed."""
    id: str
    name: str
    description: str
    input_data: Any
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    assigned_to: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """Step in a workflow."""
    name: str
    agent_role: AgentRole
    task_template: dict
    dependencies: list[str] = field(default_factory=list)
    condition: Optional[str] = None


class Agent(ABC):
    """Base agent class."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        capabilities: list[str] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities = capabilities or []
        self.status = AgentStatus.IDLE
        self.memory: dict = {}
        self.message_queue: list[AgentMessage] = []

    @abstractmethod
    async def execute(self, task: Task) -> Any:
        """Execute a task."""
        pass

    async def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.message_queue.append(message)

    async def send_message(self, orchestrator: 'Orchestrator', receiver_id: str, content: Any):
        """Send a message to another agent."""
        message = AgentMessage(
            id=hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content
        )
        await orchestrator.route_message(message)

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "tasks_completed": len([t for t in self.memory.get("tasks", []) if t.get("status") == "completed"])
        }


class WorkerAgent(Agent):
    """Worker agent that executes specific tasks."""

    def __init__(self, agent_id: str, name: str, capabilities: list[str] = None):
        super().__init__(agent_id, name, AgentRole.WORKER, capabilities)
        self.handler: Optional[Callable] = None

    def set_handler(self, handler: Callable):
        """Set the task handler function."""
        self.handler = handler

    async def execute(self, task: Task) -> Any:
        """Execute a task."""
        self.status = AgentStatus.BUSY
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        try:
            if self.handler:
                if inspect.iscoroutinefunction(self.handler):
                    result = await self.handler(task)
                else:
                    result = self.handler(task)
            else:
                result = f"Worker {self.name} completed task: {task.name}"

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self.status = AgentStatus.IDLE
            return result

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            self.status = AgentStatus.ERROR
            raise


class SpecialistAgent(Agent):
    """Specialist agent with domain expertise."""

    def __init__(self, agent_id: str, name: str, domain: str, expertise: list[str] = None):
        super().__init__(agent_id, name, AgentRole.SPECIALIST, expertise)
        self.domain = domain
        self.knowledge_base: dict = {}

    async def execute(self, task: Task) -> Any:
        """Execute a task using domain expertise."""
        self.status = AgentStatus.BUSY
        task.status = TaskStatus.RUNNING

        try:
            # Check if task is within domain
            if not self._can_handle(task):
                raise ValueError(f"Task outside specialist domain: {self.domain}")

            result = await self._process_task(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            self.status = AgentStatus.IDLE
            return result

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            self.status = AgentStatus.ERROR
            raise

    def _can_handle(self, task: Task) -> bool:
        """Check if specialist can handle the task."""
        task_text = f"{task.name} {task.description}".lower()
        return any(cap.lower() in task_text for cap in self.capabilities)

    async def _process_task(self, task: Task) -> Any:
        """Process task with domain knowledge."""
        return f"Specialist {self.name} ({self.domain}) processed: {task.name}"


class CoordinatorAgent(Agent):
    """Coordinator agent that manages other agents."""

    def __init__(self, agent_id: str, name: str):
        super().__init__(agent_id, name, AgentRole.COORDINATOR)
        self.managed_agents: list[str] = []

    async def execute(self, task: Task) -> Any:
        """Coordinate task execution across agents."""
        self.status = AgentStatus.BUSY
        task.status = TaskStatus.RUNNING

        try:
            subtasks = self._decompose_task(task)
            results = []

            for subtask in subtasks:
                result = await self._delegate_subtask(subtask)
                results.append(result)

            task.result = results
            task.status = TaskStatus.COMPLETED
            self.status = AgentStatus.IDLE
            return results

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            self.status = AgentStatus.ERROR
            raise

    def _decompose_task(self, task: Task) -> list[dict]:
        """Decompose task into subtasks."""
        return [{"name": f"subtask_{i}", "data": task.input_data}
                for i in range(3)]

    async def _delegate_subtask(self, subtask: dict) -> Any:
        """Delegate subtask to managed agent."""
        return f"Coordinated: {subtask['name']}"


class Orchestrator:
    """
    Multi-agent orchestrator.

    Manages agents, distributes tasks, and coordinates execution.
    """

    def __init__(self, name: str = "orchestrator"):
        self.name = name
        self.agents: dict[str, Agent] = {}
        self.tasks: dict[str, Task] = {}
        self.workflows: dict[str, list[WorkflowStep]] = {}
        self.message_bus: list[AgentMessage] = []
        self.shared_state: dict[str, Any] = {}
        self._execution_history: list[dict] = []

    # === Agent Management ===

    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        print(f"Registered agent: {agent.name} ({agent.role.value})")

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        self.agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def find_agents_by_role(self, role: AgentRole) -> list[Agent]:
        """Find agents by role."""
        return [a for a in self.agents.values() if a.role == role]

    def find_agents_by_capability(self, capability: str) -> list[Agent]:
        """Find agents with specific capability."""
        return [
            a for a in self.agents.values()
            if capability in a.capabilities
        ]

    def get_available_agents(self) -> list[Agent]:
        """Get all idle agents."""
        return [a for a in self.agents.values() if a.status == AgentStatus.IDLE]

    # === Task Management ===

    def create_task(
        self,
        name: str,
        description: str,
        input_data: Any = None,
        priority: int = 0,
        dependencies: list[str] = None
    ) -> Task:
        """Create a new task."""
        task_id = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:8]

        task = Task(
            id=task_id,
            name=name,
            description=description,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or []
        )

        self.tasks[task_id] = task
        return task

    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        task = self.tasks.get(task_id)
        agent = self.agents.get(agent_id)

        if not task or not agent:
            return False

        if agent.status != AgentStatus.IDLE:
            return False

        task.assigned_to = agent_id
        return True

    async def execute_task(self, task_id: str) -> Any:
        """Execute a task."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # Find or use assigned agent
        agent = None
        if task.assigned_to:
            agent = self.agents.get(task.assigned_to)
        else:
            agent = self._find_best_agent(task)
            if agent:
                task.assigned_to = agent.agent_id

        if not agent:
            raise ValueError("No suitable agent found")

        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                raise ValueError(f"Dependency not met: {dep_id}")

        # Execute
        try:
            result = await agent.execute(task)
            self._record_execution(task, agent, "success")
            return result
        except Exception as e:
            self._record_execution(task, agent, "error", str(e))
            raise

    async def execute_task_async(self, task_id: str) -> Any:
        """Execute a task asynchronously."""
        return await self.execute_task(task_id)

    def _find_best_agent(self, task: Task) -> Optional[Agent]:
        """Find the best agent for a task."""
        available = self.get_available_agents()
        if not available:
            return None

        # Simple scoring: prefer workers, then by capability match
        scored = []
        for agent in available:
            score = 0
            if agent.role == AgentRole.WORKER:
                score += 10
            elif agent.role == AgentRole.SPECIALIST:
                score += 5

            # Check capability match
            task_text = f"{task.name} {task.description}".lower()
            for cap in agent.capabilities:
                if cap.lower() in task_text:
                    score += 3

            scored.append((agent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    # === Workflow Management ===

    def define_workflow(self, name: str, steps: list[WorkflowStep]):
        """Define a workflow."""
        self.workflows[name] = steps

    async def execute_workflow(self, workflow_name: str, initial_input: Any = None) -> Any:
        """Execute a workflow."""
        steps = self.workflows.get(workflow_name)
        if not steps:
            raise ValueError(f"Workflow not found: {workflow_name}")

        current_input = initial_input
        results = []

        for step in steps:
            # Check dependencies
            deps_met = all(
                any(r.get("step") == dep for r in results)
                for dep in step.dependencies
            )

            if not deps_met:
                continue

            # Create and execute task
            task = self.create_task(
                name=step.name,
                description=f"Workflow step: {step.name}",
                input_data=current_input
            )

            # Find agent by role
            agents = self.find_agents_by_role(step.agent_role)
            if agents:
                self.assign_task(task.id, agents[0].agent_id)
                result = await self.execute_task(task.id)
                results.append({"step": step.name, "result": result})
                current_input = result

        return results

    # === Communication ===

    async def route_message(self, message: AgentMessage):
        """Route a message between agents."""
        self.message_bus.append(message)

        receiver = self.agents.get(message.receiver_id)
        if receiver:
            await receiver.receive_message(message)

    async def broadcast(self, sender_id: str, content: Any):
        """Broadcast a message to all agents."""
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                message = AgentMessage(
                    id=hashlib.md5(f"{time.time()}_{agent_id}".encode()).hexdigest()[:8],
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    content=content,
                    message_type="broadcast"
                )
                await agent.receive_message(message)

    # === Shared State ===

    def set_state(self, key: str, value: Any):
        """Set shared state."""
        self.shared_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get shared state."""
        return self.shared_state.get(key, default)

    # === Execution Tracking ===

    def _record_execution(self, task: Task, agent: Agent, status: str, error: str = None):
        """Record execution history."""
        self._execution_history.append({
            "task_id": task.id,
            "task_name": task.name,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "status": status,
            "error": error,
            "timestamp": time.time()
        })

    def get_execution_history(self) -> list[dict]:
        """Get execution history."""
        return self._execution_history

    # === Statistics ===

    def get_stats(self) -> dict:
        """Get orchestrator statistics."""
        return {
            "total_agents": len(self.agents),
            "agents_by_role": {
                role.value: len(self.find_agents_by_role(role))
                for role in AgentRole
            },
            "available_agents": len(self.get_available_agents()),
            "total_tasks": len(self.tasks),
            "tasks_by_status": {
                status.value: len([t for t in self.tasks.values() if t.status == status])
                for status in TaskStatus
            },
            "total_messages": len(self.message_bus),
            "workflows": len(self.workflows),
            "shared_state_keys": list(self.shared_state.keys())
        }


class WorkflowBuilder:
    """Builder for creating workflows."""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.steps: list[WorkflowStep] = []

    def add_step(
        self,
        name: str,
        role: AgentRole,
        task_template: dict,
        dependencies: list[str] = None,
        condition: str = None
    ) -> 'WorkflowBuilder':
        """Add a workflow step."""
        step = WorkflowStep(
            name=name,
            agent_role=role,
            task_template=task_template,
            dependencies=dependencies or [],
            condition=condition
        )
        self.steps.append(step)
        return self

    def build(self, workflow_name: str):
        """Build and register the workflow."""
        self.orchestrator.define_workflow(workflow_name, self.steps)
        return self.orchestrator


class TaskDecomposer:
    """Decompose complex tasks into subtasks."""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator

    def decompose(self, task: Task, strategy: str = "sequential") -> list[Task]:
        """Decompose a task into subtasks."""
        if strategy == "sequential":
            return self._decompose_sequential(task)
        elif strategy == "parallel":
            return self._decompose_parallel(task)
        elif strategy == "hierarchical":
            return self._decompose_hierarchical(task)
        return [task]

    def _decompose_sequential(self, task: Task) -> list[Task]:
        """Decompose into sequential subtasks."""
        subtasks = []
        steps = task.description.split("\n") if "\n" in task.description else [task.description]

        for i, step in enumerate(steps):
            subtask = self.orchestrator.create_task(
                name=f"{task.name}_step_{i+1}",
                description=step.strip(),
                input_data=task.input_data,
                dependencies=[subtasks[-1].id] if subtasks else []
            )
            subtasks.append(subtask)

        return subtasks

    def _decompose_parallel(self, task: Task) -> list[Task]:
        """Decompose into parallel subtasks."""
        subtasks = []
        steps = task.description.split("\n") if "\n" in task.description else [task.description]

        for i, step in enumerate(steps):
            subtask = self.orchestrator.create_task(
                name=f"{task.name}_part_{i+1}",
                description=step.strip(),
                input_data=task.input_data
            )
            subtasks.append(subtask)

        return subtasks

    def _decompose_hierarchical(self, task: Task) -> list[Task]:
        """Decompose hierarchically."""
        main_task = self.orchestrator.create_task(
            name=f"{task.name}_main",
            description=f"Coordinate: {task.description}",
            input_data=task.input_data
        )

        subtask1 = self.orchestrator.create_task(
            name=f"{task.name}_analyze",
            description=f"Analyze: {task.description}",
            dependencies=[main_task.id]
        )

        subtask2 = self.orchestrator.create_task(
            name=f"{task.name}_execute",
            description=f"Execute: {task.description}",
            dependencies=[subtask1.id]
        )

        return [main_task, subtask1, subtask2]


# === Convenience Functions ===

def create_orchestrator(name: str = "orchestrator") -> Orchestrator:
    """Create a new orchestrator."""
    return Orchestrator(name)

def create_worker(name: str, capabilities: list[str] = None) -> WorkerAgent:
    """Create a worker agent."""
    agent_id = hashlib.md5(f"worker_{name}_{time.time()}".encode()).hexdigest()[:8]
    return WorkerAgent(agent_id, name, capabilities)

def create_specialist(name: str, domain: str, expertise: list[str] = None) -> SpecialistAgent:
    """Create a specialist agent."""
    agent_id = hashlib.md5(f"specialist_{name}_{time.time()}".encode()).hexdigest()[:8]
    return SpecialistAgent(agent_id, name, domain, expertise)

async def demo():
    """Demo the orchestrator."""
    orchestrator = create_orchestrator("demo-orchestrator")

    # Create agents
    worker1 = create_worker("researcher", ["research", "search", "analyze"])
    worker2 = create_worker("coder", ["code", "programming", "implementation"])
    specialist = create_specialist("data-scientist", "data-science", ["ml", "analytics", "data"])

    orchestrator.register_agent(worker1)
    orchestrator.register_agent(worker2)
    orchestrator.register_agent(specialist)

    # Create and execute task
    task = orchestrator.create_task(
        name="research-and-implement",
        description="Research best practices\nAnalyze requirements\nImplement solution",
        input_data={"topic": "machine learning"}
    )

    result = await orchestrator.execute_task(task.id)
    print(f"Result: {result}")
    print(f"Stats: {orchestrator.get_stats()}")
