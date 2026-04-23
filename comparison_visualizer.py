import matplotlib.pyplot as plt


def extract_best_fitness(history):
    return [item["best_fitness"] for item in history]


def extract_avg_fitness(history):
    return [item["average_fitness"] for item in history]


def extract_generations(history):
    return [item["generation"] for item in history]


def comparison_visualizer(no_evolution_history, random_history, llm_history):
    no_gen = extract_generations(no_evolution_history)
    rand_gen = extract_generations(random_history)
    llm_gen = extract_generations(llm_history)

    no_best = extract_best_fitness(no_evolution_history)
    rand_best = extract_best_fitness(random_history)
    llm_best = extract_best_fitness(llm_history)

    plt.figure(figsize=(10, 6))
    plt.plot(no_gen, no_best, marker="o", label="No Evolution")
    plt.plot(rand_gen, rand_best, marker="s", label="Random Mutation")
    plt.plot(llm_gen, llm_best, marker="^", label="LLM-Guided Mutation")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Comparison of Fitness Progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()