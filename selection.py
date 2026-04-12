from evolution_engine import Candidate, EvaluationResult


def select_candidates(population, evaluated, config):
    """
    Engine-compatible selection with logging and diversity handling.
    """

    if len(population) != len(evaluated):
        raise ValueError("Population and evaluation size mismatch")

    k = config.get("keep_elites", 3)

    # Pair candidates with evaluation results
    candidates = list(zip(population, evaluated))

    # Keep only passed candidates
    candidates = [pair for pair in candidates if pair[1].passed]

    if not candidates:
        print(" No valid candidates — falling back to previous population")

    # Fallback: return top-k from original population (even if invalid)
    fallback = population[:k]

    for i, candidate in enumerate(fallback):
        print(
            f"Fallback Selected Rank {i+1} | "
            f"Candidate ID: {candidate.id}"
        )

        return fallback

    # Sort by fitness descending
    candidates.sort(key=lambda x: x[1].fitness, reverse=True)

    print("\n=== Selection Phase ===")

    selected_population = []
    seen = set()

    for i, (candidate, result) in enumerate(candidates):
        if len(selected_population) >= k:
            break

        # Avoid duplicate solutions
        solution_key = candidate.solution.strip()

        if solution_key not in seen:
            selected_population.append(candidate)
            seen.add(solution_key)

            print(
                f"Selected Rank {i+1} | "
                f"Candidate ID: {candidate.id} | "
                f"Fitness: {result.fitness:.4f}"
            )

    # Fallback in case all top-k were duplicates
    if not selected_population:
        best_candidate, best_result = candidates[0]
        selected_population.append(best_candidate)
        print(
            f"Fallback Selected Best | "
            f"Candidate ID: {best_candidate.id} | "
            f"Fitness: {best_result.fitness:.4f}"
        )

    best_candidate, best_result = candidates[0]
    print(
        f"Best Candidate Fitness: {best_result.fitness:.4f} | "
        f"Candidate ID: {best_candidate.id}"
    )

    return selected_population