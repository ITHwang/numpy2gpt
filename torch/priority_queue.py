import heapq
from typing import Generic, TypeVar

T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """A priority queue that supports push and pop operations.

    heapq is a min-heap, so we negate the priority value to make it a max-heap.
    """

    def __init__(self) -> None:
        self._queue: list[tuple[float, int, T]] = []

    def push(self, item: T, priority: float) -> None:
        """Add an item to the queue with the given priority."""
        # NOTE: id(item) is just for breaking ties in the queue.
        # The order of items with the same priority does not matter.
        heapq.heappush(self._queue, (-priority, id(item), item))

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
