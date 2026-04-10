from initializer import EvolutionConfig, Initializer, CandidateGenerator
from engine_adapter import wrap_population, GeneratorAdapter
from evolution_engine import EvolutionEngine
from selection import select_candidates
from fitness_evaluator import fitness_evaluator
from comparison_visualizer import comparison_visualizer
from export_csv import export_to_csv
from visualizer import visualizer


def run_experiment(use_case, mode, llm_client=None):
    config_obj = EvolutionConfig()
    config = {
        "population_size": config_obj.population_size,
        "generations": config_obj.generations,
        "mutation_rate": 0.8,
        "selection_strategy": config_obj.selection_strategy,
        "keep_elites": 5,
    }

    initializer = Initializer(config_obj)
    original_generator = CandidateGenerator(llm_client=llm_client)

    base_code = initializer.get_base_algorithm(use_case)

    if mode == "none":
        population_strings = [base_code] * config_obj.population_size
    else:
        population_strings = initializer.initialize_population(
            base_code=base_code,
            generator=original_generator,
            use_case=use_case
        )

    initial_population = wrap_population(population_strings, use_case, source="initial")
    generator = GeneratorAdapter(original_generator, mode=mode)

    engine = EvolutionEngine(
        candidate_generator=generator,
        fitness_evaluator=fitness_evaluator,
        selector=select_candidates,
        visualizer=None,
        config=config,
    )

    result = engine.run(
        initial_population=initial_population,
        use_case=use_case,
        num_generations=config_obj.generations,
    )

    return result, engine.metrics_tracker


def run_full_pipeline_for_use_case(use_case):
    print(f"\n{'=' * 20} RUNNING USE CASE: {use_case} {'=' * 20}")

    no_result, no_tracker = run_experiment(use_case, mode="none")
    random_result, random_tracker = run_experiment(use_case, mode="random")
    llm_result, llm_tracker = run_experiment(use_case, mode="llm", llm_client=None)

    no_history = no_tracker.as_dict()
    random_history = random_tracker.as_dict()
    llm_history = llm_tracker.as_dict()

    # Export CSV files with use-case prefix
    export_to_csv(f"{use_case}_no_evolution.csv", no_history)
    export_to_csv(f"{use_case}_random_mutation.csv", random_history)
    export_to_csv(f"{use_case}_llm_mutation.csv", llm_history)

    print(f"\nCSV files exported successfully for {use_case}.")

    # Individual visualizations
    print(f"\n=== {use_case} | No Evolution Visualization ===")
    visualizer(no_tracker)

    print(f"\n=== {use_case} | Random Mutation Visualization ===")
    visualizer(random_tracker)

    print(f"\n=== {use_case} | LLM-Guided Mutation Visualization ===")
    visualizer(llm_tracker)

    # Final comparison for this use case
    comparison_visualizer(no_history, random_history, llm_history)

    print(f"\n=== Final Comparison for {use_case} ===")
    print("No Evolution Best Fitness:", no_result["best_fitness"])
    print("Random Mutation Best Fitness:", random_result["best_fitness"])
    print("LLM-Guided Mutation Best Fitness:", llm_result["best_fitness"])


def main():
    use_cases = ["cart_pole", "matrix_multiplication"]

    for use_case in use_cases:
        run_full_pipeline_for_use_case(use_case)


if __name__ == "__main__":
    main()
