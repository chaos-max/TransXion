"""并行处理工具函数，用于加速模型推理和Agent交互。"""
import asyncio
from typing import List, Callable, Any, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import time


class ParallelExecutor:
    """
    并行执行器，支持异步和多线程两种模式。

    使用场景：
    1. 批量模型推理
    2. 多个Agent并行调用
    3. 批量数据处理
    """

    def __init__(self, max_workers: int = 5, mode: str = "async"):
        """
        Args:
            max_workers: 最大并行数
            mode: 执行模式，'async' 或 'thread'
        """
        self.max_workers = max_workers
        self.mode = mode
        self.executor = None
        if mode == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run_async_batch(
        self,
        tasks: List[Callable],
        show_progress: bool = True,
        return_exceptions: bool = True,
    ) -> List[Any]:
        """
        异步批量执行任务。

        Args:
            tasks: 协程任务列表
            show_progress: 是否显示进度
            return_exceptions: 是否返回异常（True则不会中断）

        Returns:
            结果列表
        """
        if show_progress:
            logger.info(f"开始并行执行 {len(tasks)} 个异步任务")

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        elapsed = time.time() - start_time

        if show_progress:
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(
                f"完成 {success_count}/{len(tasks)} 个任务，"
                f"耗时 {elapsed:.2f}秒，"
                f"平均 {elapsed/len(tasks):.2f}秒/任务"
            )

        return results

    def run_thread_batch(
        self,
        func: Callable,
        args_list: List[tuple],
        show_progress: bool = True,
    ) -> List[Any]:
        """
        多线程批量执行任务。

        Args:
            func: 要执行的函数
            args_list: 参数列表，每个元素是一个tuple
            show_progress: 是否显示进度

        Returns:
            结果列表
        """
        if show_progress:
            logger.info(f"开始并行执行 {len(args_list)} 个线程任务")

        start_time = time.time()
        futures = []

        for args in args_list:
            future = self.executor.submit(func, *args)
            futures.append(future)

        results = []
        completed = 0

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if show_progress and completed % 10 == 0:
                    logger.info(f"已完成 {completed}/{len(args_list)} 个任务")
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                results.append(None)

        elapsed = time.time() - start_time

        if show_progress:
            logger.info(
                f"完成 {len(results)} 个任务，"
                f"耗时 {elapsed:.2f}秒，"
                f"平均 {elapsed/len(args_list):.2f}秒/任务"
            )

        return results

    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)


async def batch_call_agents(
    agents: List[Any],
    func_name: str,
    kwargs_list: Optional[List[Dict]] = None,
    max_parallel: int = 5,
) -> List[Any]:
    """
    批量并行调用多个Agent的同一个方法。

    Args:
        agents: Agent列表
        func_name: 要调用的方法名
        kwargs_list: 每个Agent的参数字典列表（如果为None，则所有Agent使用相同参数{}）
        max_parallel: 最大并行数

    Returns:
        结果列表

    Example:
        >>> agents = [agent1, agent2, agent3]
        >>> results = await batch_call_agents(
        ...     agents,
        ...     "process",
        ...     [{"input": "a"}, {"input": "b"}, {"input": "c"}]
        ... )
    """
    if kwargs_list is None:
        kwargs_list = [{}] * len(agents)

    assert len(agents) == len(kwargs_list), "agents和kwargs_list长度必须相同"

    async def call_single_agent(agent, kwargs):
        """调用单个Agent"""
        loop = asyncio.get_event_loop()
        func = getattr(agent, func_name)
        return await loop.run_in_executor(None, lambda: func(**kwargs))

    # 分批执行以控制并发数
    results = []
    for i in range(0, len(agents), max_parallel):
        batch_agents = agents[i:i+max_parallel]
        batch_kwargs = kwargs_list[i:i+max_parallel]

        tasks = [
            call_single_agent(agent, kwargs)
            for agent, kwargs in zip(batch_agents, batch_kwargs)
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)

    return results
