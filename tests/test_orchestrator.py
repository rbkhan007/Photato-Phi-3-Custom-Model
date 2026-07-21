"""Tests for the orchestrator package (sync logic + asyncio.run for coroutines)."""

import asyncio

from orchestrator import (
    Orchestrator, WorkerAgent, SpecialistAgent, CoordinatorAgent,
    AgentRole, AgentStatus, TaskStatus, Task, AgentMessage, WorkflowStep,
    WorkflowBuilder, TaskDecomposer,
    create_orchestrator, create_worker, create_specialist,
)


class TestFactories:
    def test_create_worker(self):
        w = create_worker("researcher", ["research"])
        assert isinstance(w, WorkerAgent)
        assert w.role == AgentRole.WORKER
        assert w.capabilities == ["research"]
        assert w.status == AgentStatus.IDLE

    def test_create_specialist(self):
        s = create_specialist("ds", "data", ["ml"])
        assert isinstance(s, SpecialistAgent)
        assert s.domain == "data"
        assert s.role == AgentRole.SPECIALIST


class TestOrchestratorManagement:
    def test_register_and_find(self):
        orch = create_orchestrator()
        w = create_worker("w1", ["code"])
        orch.register_agent(w)
        assert orch.get_agent(w.agent_id) is w
        assert orch.find_agents_by_role(AgentRole.WORKER) == [w]
        assert orch.find_agents_by_capability("code") == [w]
        assert orch.get_available_agents() == [w]

    def test_unregister(self):
        orch = create_orchestrator()
        w = create_worker("w1")
        orch.register_agent(w)
        orch.unregister_agent(w.agent_id)
        assert orch.get_agent(w.agent_id) is None

    def test_create_and_assign_task(self):
        orch = create_orchestrator()
        w = create_worker("w1")
        orch.register_agent(w)
        task = orch.create_task("t", "desc", input_data={"x": 1})
        assert isinstance(task, Task)
        assert task.status == TaskStatus.PENDING
        assert orch.assign_task(task.id, w.agent_id) is True
        assert task.assigned_to == w.agent_id

    def test_assign_task_bad_ids(self):
        orch = create_orchestrator()
        assert orch.assign_task("nope", "nope") is False


class TestExecution:
    def test_worker_execute_default_handler(self):
        orch = create_orchestrator()
        w = create_worker("w1")
        orch.register_agent(w)
        task = orch.create_task("job", "do it")
        result = asyncio.run(orch.execute_task(task.id))
        assert "completed task" in result
        assert task.status == TaskStatus.COMPLETED
        assert task.assigned_to == w.agent_id

    def test_worker_custom_handler(self):
        w = create_worker("w1")
        w.set_handler(lambda t: f"handled:{t.name}")
        task = Task(id="1", name="mytask", description="d", input_data=None)
        result = asyncio.run(w.execute(task))
        assert result == "handled:mytask"
        assert task.status == TaskStatus.COMPLETED

    def test_worker_handler_failure(self):
        w = create_worker("w1")
        def boom(t):
            raise RuntimeError("fail")
        w.set_handler(boom)
        task = Task(id="1", name="t", description="d", input_data=None)
        try:
            asyncio.run(w.execute(task))
        except RuntimeError:
            pass
        assert task.status == TaskStatus.FAILED
        assert w.status == AgentStatus.ERROR

    def test_specialist_can_handle(self):
        s = create_specialist("s", "data", ["ml", "analytics"])
        in_domain = Task(id="1", name="ml pipeline", description="build ml", input_data=None)
        out_domain = Task(id="2", name="cooking", description="make food", input_data=None)
        assert s._can_handle(in_domain) is True
        assert s._can_handle(out_domain) is False

    def test_specialist_execute(self):
        s = create_specialist("s", "data", ["ml"])
        task = Task(id="1", name="ml job", description="ml work", input_data=None)
        result = asyncio.run(s.execute(task))
        assert "Specialist" in result
        assert task.status == TaskStatus.COMPLETED

    def test_coordinator_execute(self):
        c = CoordinatorAgent("c1", "coord")
        task = Task(id="1", name="big", description="d", input_data="in")
        result = asyncio.run(c.execute(task))
        assert isinstance(result, list)
        assert len(result) == 3

    def test_find_best_agent_prefers_worker(self):
        orch = create_orchestrator()
        w = create_worker("w", ["code"])
        s = create_specialist("s", "data", ["ml"])
        orch.register_agent(w)
        orch.register_agent(s)
        task = orch.create_task("code task", "write code")
        best = orch._find_best_agent(task)
        assert best is w

    def test_execute_task_dependency_not_met(self):
        orch = create_orchestrator()
        w = create_worker("w")
        orch.register_agent(w)
        dep = orch.create_task("dep", "d")
        task = orch.create_task("main", "d", dependencies=[dep.id])
        orch.assign_task(task.id, w.agent_id)
        try:
            asyncio.run(orch.execute_task(task.id))
            assert False, "expected ValueError"
        except ValueError as e:
            assert "Dependency not met" in str(e)


class TestCommunicationAndState:
    def test_route_message(self):
        orch = create_orchestrator()
        w = create_worker("w1")
        orch.register_agent(w)
        msg = AgentMessage(id="m1", sender_id="x", receiver_id=w.agent_id, content="hi")
        asyncio.run(orch.route_message(msg))
        assert msg in orch.message_bus
        assert w.message_queue[-1] is msg

    def test_broadcast(self):
        orch = create_orchestrator()
        a = create_worker("a")
        b = create_worker("b")
        orch.register_agent(a)
        orch.register_agent(b)
        asyncio.run(orch.broadcast(a.agent_id, "hello"))
        assert len(a.message_queue) == 0
        assert len(b.message_queue) == 1

    def test_shared_state(self):
        orch = create_orchestrator()
        orch.set_state("k", 42)
        assert orch.get_state("k") == 42
        assert orch.get_state("missing", "def") == "def"

    def test_stats(self):
        orch = create_orchestrator()
        orch.register_agent(create_worker("w"))
        orch.create_task("t", "d")
        stats = orch.get_stats()
        assert stats["total_agents"] == 1
        assert stats["total_tasks"] == 1
        assert stats["agents_by_role"]["worker"] == 1


class TestWorkflowAndDecompose:
    def test_workflow_builder(self):
        orch = create_orchestrator()
        builder = WorkflowBuilder(orch)
        result = (builder
                  .add_step("s1", AgentRole.WORKER, {"a": 1})
                  .add_step("s2", AgentRole.WORKER, {"b": 2}, dependencies=["s1"])
                  .build("wf"))
        assert result is orch
        assert len(orch.workflows["wf"]) == 2
        assert isinstance(orch.workflows["wf"][0], WorkflowStep)

    def test_execute_workflow(self):
        orch = create_orchestrator()
        w = create_worker("w")
        orch.register_agent(w)
        WorkflowBuilder(orch).add_step("step1", AgentRole.WORKER, {}).build("wf")
        results = asyncio.run(orch.execute_workflow("wf", initial_input="start"))
        assert results[0]["step"] == "step1"

    def test_decompose_sequential(self):
        orch = create_orchestrator()
        dec = TaskDecomposer(orch)
        task = orch.create_task("multi", "step a\nstep b\nstep c")
        subtasks = dec.decompose(task, strategy="sequential")
        assert len(subtasks) == 3
        assert subtasks[1].dependencies == [subtasks[0].id]

    def test_decompose_parallel(self):
        orch = create_orchestrator()
        dec = TaskDecomposer(orch)
        task = orch.create_task("multi", "a\nb")
        subtasks = dec.decompose(task, strategy="parallel")
        assert len(subtasks) == 2
        assert all(st.dependencies == [] for st in subtasks)

    def test_decompose_hierarchical(self):
        orch = create_orchestrator()
        dec = TaskDecomposer(orch)
        task = orch.create_task("root", "do stuff")
        subtasks = dec.decompose(task, strategy="hierarchical")
        assert len(subtasks) == 3
