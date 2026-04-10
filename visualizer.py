import matplotlib.pyplot as plt


def visualizer(metrics_tracker):
    history = metrics_tracker.history

    if not history:
        print("No history available to visualize.")
        return

    generations = [h.generation for h in history]
    best_fitness = [h.best_fitness for h in history]
    avg_fitness = [h.average_fitness for h in history]
    variance_fitness = [h.variance_fitness for h in history]
    duration_sec = [h.duration_sec for h in history]
    steps_per_generation = [h.steps_per_generation for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(generations, best_fitness, marker="o", label="Best Fitness")
    plt.plot(generations, avg_fitness, marker="s", label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Progression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(generations, variance_fitness, marker="^", label="Fitness Variance")
    plt.xlabel("Generation")
    plt.ylabel("Variance")
    plt.title("Fitness Variance Across Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(generations, duration_sec, marker="d", label="Duration (sec)")
    plt.xlabel("Generation")
    plt.ylabel("Seconds")
    plt.title("Generation Execution Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(generations, steps_per_generation, marker="x", label="Steps per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Steps")
    plt.title("Steps per Generation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    latest = history[-1]
    print("\n=== Evolution Summary ===")
    print(f"Last Generation: {latest.generation}")
    print(f"Best Fitness: {latest.best_fitness:.4f}")
    print(f"Average Fitness: {latest.average_fitness:.4f}")
    print(f"Variance: {latest.variance_fitness:.4f}")
    print(f"Population Size: {latest.population_size}")
    print(f"Duration: {latest.duration_sec:.4f} sec")
    print(f"Steps per Generation: {latest.steps_per_generation}")

    if metrics_tracker.best_overall_candidate is not None:
        print(f"Best Overall Candidate ID: {metrics_tracker.best_overall_candidate.id}")
        print(f"Best Overall Fitness: {metrics_tracker.best_overall_fitness:.4f}")