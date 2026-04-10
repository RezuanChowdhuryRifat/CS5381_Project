import csv
import os

def export_to_csv(filename, history):
    print("Saving CSV to:", os.path.abspath(filename))

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            "generation",
            "best_fitness",
            "average_fitness",
            "variance_fitness",
            "duration_sec",
            "population_size",
            "steps_per_generation"
        ])

        for item in history:
            writer.writerow([
                item.get("generation"),
                item.get("best_fitness"),
                item.get("average_fitness"),
                item.get("variance_fitness"),
                item.get("duration_sec"),
                item.get("population_size"),
                item.get("steps_per_generation")
            ])

    print(f"{filename} saved successfully!")