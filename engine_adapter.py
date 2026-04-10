import uuid
import random
from typing import List, Dict, Any

from evolution_engine import Candidate


def wrap_population(population_strings: List[str], use_case: str, source="initial") -> List[Candidate]:
    candidates = []

    for code in population_strings:
        candidates.append(
            Candidate(
                id=f"cand_{uuid.uuid4().hex[:8]}",
                solution=code,
                source=source,
                metadata={"use_case": use_case}
            )
        )

    return candidates


class GeneratorAdapter:
    def __init__(self, original_generator, mode="random"):
        self.original_generator = original_generator
        self.mode = mode

    def __call__(
        self,
        selected_candidates: List[Candidate],
        config: Dict[str, Any],
        generation: int,
        use_case: str,
    ) -> List[Candidate]:

        if not selected_candidates:
            raise ValueError("No selected candidates provided")

        population_size = config.get("population_size", 10)
        new_population: List[Candidate] = []

        while len(new_population) < population_size:
            parent = random.choice(selected_candidates)

            if self.mode == "none":
                mutated_code = parent.solution
                source = "none"

            elif self.mode == "llm":
                mutated_code = self.original_generator.llm_guided_mutation(
                    current_code=parent.solution,
                    use_case=use_case,
                    fitness_feedback="Improve fitness based on previous evaluation."
                )
                source = "llm"

            else:
                mutated_code = self.original_generator.random_mutation(
                    parent.solution,
                    use_case
                )
                source = "random"

            new_population.append(
                Candidate(
                    id=f"cand_{uuid.uuid4().hex[:8]}",
                    solution=mutated_code,
                    source=source,
                    metadata={
                        "parent_id": parent.id,
                        "generation": generation + 1,
                        "use_case": use_case
                    }
                )
            )

        return new_population