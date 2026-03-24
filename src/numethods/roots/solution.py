from dataclasses import dataclass, field


@dataclass(slots=True)
class RootIteration:
    i: int
    sol: float
    ea: float


@dataclass(slots=True)
class RootResult:
    solution: float = float("nan")
    relative_error: float = float("inf")
    success: bool = False
    status: int = -1
    message: str = ""
    nfev: int = 0
    history: list[RootIteration] = field(default_factory=list)

    @property
    def nit(self) -> int:
        return len(self.history)

    def add_iter(self, sol: float, ea: float) -> RootIteration:
        self.solution = sol
        self.relative_error = ea
        iteration = RootIteration(i=self.nit + 1, sol=sol, ea=ea)
        self.history.append(iteration)
        return iteration

    def add_feval(self, count: int = 1) -> int:
        if count < 0:
            raise ValueError("'count' must be >= 0.")
        self.nfev += count
        return self.nfev
