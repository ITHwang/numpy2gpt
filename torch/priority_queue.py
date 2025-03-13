import heapq
from typing import Generic, TypeVar

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    def __init__(self) -> None:
        self._queue: list[tuple[float, int, T]] = []
        self._index: int = 0

    def push(self, item: T, priority: float) -> None:
        """Add an item to the queue with the given priority."""
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self) -> T:
        """Remove and return the item with the lowest priority value."""
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        return heapq.heappop(self._queue)[-1]

    def peek(self) -> T | None:
        """Return the item with the lowest priority without removing it."""
        if self.is_empty():
            return None
        return self._queue[0][-1]

    def is_empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self._queue)
