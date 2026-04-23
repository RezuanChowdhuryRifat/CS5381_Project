from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
import time
import statistics


@dataclass
class Candidate:
    id: str
    solution: Any
    source: str = "random"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    candidate_id: str
    fitness: float
    metrics: Dict[str, float]
    passed: bool = True
    error: Optional[str] = None


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    average_fitness: float
    variance_fitness: float
    best_candidate_id: str
    duration_sec: float
    population_size: int
    steps_per_generation: int


class MetricsTracker:
    def __init__(self) -> None:
        self.history: List[GenerationStats] = []
        self.best_overall_candidate: Optional[Candidate] = None
        self.best_overall_fitness: float = float("-inf")

    def update(
        self,
        generation: int,
        evaluated: List[EvaluationResult],
        population: List[Candidate],
        duration_sec: float
    ) -> None:
        passed_results = [r for r in evaluated if r.passed]
        fitness_values = [r.fitness for r in passed_results]

        steps_per_generation = sum(
            int(r.metrics.get("steps", 0)) for r in passed_results
        )

        if not fitness_values:
            stats = GenerationStats(
                generation=generation,
                best_fitness=float("-inf"),
                average_fitness=float("-inf"),
                variance_fitness=0.0,
                best_candidate_id="NONE",
                duration_sec=duration_sec,
                population_size=len(population),
                steps_per_generation=0,
            )
            self.history.append(stats)
            return

        best_result = max(passed_results, key=lambda x: x.fitness)
        avg_fit = statistics.mean(fitness_values)
        var_fit = statistics.pvariance(fitness_values) if len(fitness_values) > 1 else 0.0

        stats = GenerationStats(
            generation=generation,
            best_fitness=best_result.fitness,
            average_fitness=avg_fit,
            variance_fitness=var_fit,
            best_candidate_id=best_result.candidate_id,
            duration_sec=duration_sec,
            population_size=len(population),
            steps_per_generation=steps_per_generation,
        )
        self.history.append(stats)

        if best_result.fitness > self.best_overall_fitness:
            self.best_overall_fitness = best_result.fitness
            best_candidate = next(
                (c for c in population if c.id == best_result.candidate_id),
                None
            )
            self.best_overall_candidate = best_candidate

    def as_dict(self) -> List[Dict[str, Any]]:
        return [vars(h) for h in self.history]


class EvolutionEngine:
    def __init__(
        self,
        candidate_generator: Callable[..., List[Candidate]],
        fitness_evaluator: Callable[[Candidate, str], EvaluationResult],
        selector: Callable[[List[Candidate], List[EvaluationResult], Dict[str, Any]], List[Candidate]],
        visualizer: Optional[Callable[[MetricsTracker], None]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.fitness_evaluator = fitness_evaluator
        self.selector = selector
        self.visualizer = visualizer
        self.config = config or {}
        self.metrics_tracker = MetricsTracker()

    def evaluate_population(self, population: List[Candidate], use_case: str) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []

        for candidate in population:
            try:
                result = self.fitness_evaluator(candidate, use_case)
                results.append(result)
            except Exception as e:
                results.append(
                    EvaluationResult(
                        candidate_id=candidate.id,
                        fitness=float("-inf"),
                        metrics={},
                        passed=False,
                        error=str(e),
                    )
                )

        return results

    def run(
        self,
        initial_population: List[Candidate],
        use_case: str,
        num_generations: int,
    ) -> Dict[str, Any]:
        population = initial_population

        for generation in range(num_generations):
            start_time = time.time()

            evaluated = self.evaluate_population(population, use_case)
            selected = self.selector(population, evaluated, self.config)

            next_population = self.candidate_generator(
                selected_candidates=selected,
                config=self.config,
                generation=generation,
                use_case=use_case,
            )

            duration_sec = time.time() - start_time

            self.metrics_tracker.update(
                generation=generation,
                evaluated=evaluated,
                population=population,
                duration_sec=duration_sec,
            )

            if self.visualizer is not None:
                self.visualizer(self.metrics_tracker)

            population = next_population

        return {
            "best_candidate": self.metrics_tracker.best_overall_candidate,
            "best_fitness": self.metrics_tracker.best_overall_fitness,
            "history": self.metrics_tracker.as_dict(),
            "final_population": population,
        }