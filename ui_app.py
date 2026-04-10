import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

from main import run_experiment
from initializer import EvolutionConfig, Initializer
from export_csv import export_to_csv


st.set_page_config(page_title="Group-8 CS5381 Evolution UI", layout="wide")


USE_CASE_DESCRIPTIONS = {
    "cart_pole": """
CartPole control problem:
The candidate program must decide whether to push left or right based on the cart-pole observation.
The evaluation considers survival time, pole-angle stability, and cart-position drift.
""",
    "matrix_multiplication": """
3x3 Matrix Multiplication:
The candidate program must correctly multiply two 3x3 matrices.
The evaluation considers correctness and operation cost such as multiplications and additions.
"""
}


def plot_fitness(history, title):
    generations = [h["generation"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    avg_fitness = [h["average_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(generations, best_fitness, marker="o", label="Best Fitness")
    ax.plot(generations, avg_fitness, marker="s", label="Average Fitness")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_variance(history, title):
    generations = [h["generation"] for h in history]
    variance = [h["variance_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(generations, variance, marker="^", label="Fitness Variance")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Variance")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_runtime(history, title):
    generations = [h["generation"] for h in history]
    durations = [h["duration_sec"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(generations, durations, marker="d", label="Duration (sec)")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Seconds")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_steps(history, title):
    generations = [h["generation"] for h in history]
    steps = [h["steps_per_generation"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(generations, steps, marker="x", label="Steps per Generation")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Steps")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_comparison(no_history, random_history, llm_history):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        [h["generation"] for h in no_history],
        [h["best_fitness"] for h in no_history],
        marker="o",
        label="No Evolution"
    )
    ax.plot(
        [h["generation"] for h in random_history],
        [h["best_fitness"] for h in random_history],
        marker="s",
        label="Random Mutation"
    )
    ax.plot(
        [h["generation"] for h in llm_history],
        [h["best_fitness"] for h in llm_history],
        marker="^",
        label="LLM-Guided Mutation"
    )

    ax.set_title("Comparison of Fitness Progression")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def run_with_logs(use_case, mode, llm_client=None):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result, tracker = run_experiment(use_case, mode, llm_client=llm_client)
    logs = buffer.getvalue()
    history = tracker.as_dict()
    return result, tracker, history, logs


st.title("CS5381 Evolutionary Algorithm UI")

with st.sidebar:
    st.header("Configuration")
    use_case = st.selectbox(
        "Use Case",
        ["cart_pole", "matrix_multiplication"],
        index=0
    )
    run_button = st.button("Run Experiments")

config_obj = EvolutionConfig()
initializer = Initializer(config_obj)
base_code = initializer.get_base_algorithm(use_case)

st.subheader("Algorithm / Problem Description")
st.write(USE_CASE_DESCRIPTIONS[use_case])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Initial Code")
    st.code(base_code, language="python")

with col2:
    st.subheader("Experiment Settings")
    st.write(f"Population Size: {config_obj.population_size}")
    st.write(f"Generations: {config_obj.generations}")
    st.write(f"Mutation Rate: {config_obj.mutation_rate}")
    st.write(f"Selection Strategy: {config_obj.selection_strategy}")

if use_case == "matrix_multiplication":
    st.warning("Matrix mode is still experimental. For final submission, CartPole is the stable prototype.")

if run_button:
    st.info("Running experiments...")

    # Paste your real API key 
    llm_client = OpenAI(api_key="Enter-your-API-Key")

    no_result, no_tracker, no_history, no_logs = run_with_logs(use_case, mode="none", llm_client=None)
    random_result, random_tracker, random_history, random_logs = run_with_logs(use_case, mode="random", llm_client=None)
    llm_result, llm_tracker, llm_history, llm_logs = run_with_logs(use_case, mode="llm", llm_client=llm_client)

    export_to_csv(f"{use_case}_no_evolution.csv", no_history)
    export_to_csv(f"{use_case}_random_mutation.csv", random_history)
    export_to_csv(f"{use_case}_llm_mutation.csv", llm_history)

    st.success("CSV files exported successfully.")

    st.subheader("Final Best Scores")
    score_col1, score_col2, score_col3 = st.columns(3)
    score_col1.metric("No Evolution", f"{no_result['best_fitness']:.4f}")
    score_col2.metric("Random Mutation", f"{random_result['best_fitness']:.4f}")
    score_col3.metric("LLM-Guided Mutation", f"{llm_result['best_fitness']:.4f}")

    st.subheader("Comparison Plot")
    st.pyplot(plot_comparison(no_history, random_history, llm_history))

    tab1, tab2, tab3 = st.tabs(["No Evolution", "Random Mutation", "LLM-Guided Mutation"])

    with tab1:
        st.write("### Final Best Solution")
        if no_result["best_candidate"] is not None:
            st.code(no_result["best_candidate"].solution, language="python")

        st.write("### Fitness Progression")
        st.pyplot(plot_fitness(no_history, "No Evolution - Fitness Progression"))

        st.write("### Fitness Variance")
        st.pyplot(plot_variance(no_history, "No Evolution - Fitness Variance"))

        st.write("### Runtime")
        st.pyplot(plot_runtime(no_history, "No Evolution - Generation Execution Time"))

        st.write("### Steps")
        st.pyplot(plot_steps(no_history, "No Evolution - Steps per Generation"))

        st.write("### Operation Logs")
        st.text(no_logs if no_logs.strip() else "No logs available.")

    with tab2:
        st.write("### Final Best Solution")
        if random_result["best_candidate"] is not None:
            st.code(random_result["best_candidate"].solution, language="python")

        st.write("### Fitness Progression")
        st.pyplot(plot_fitness(random_history, "Random Mutation - Fitness Progression"))

        st.write("### Fitness Variance")
        st.pyplot(plot_variance(random_history, "Random Mutation - Fitness Variance"))

        st.write("### Runtime")
        st.pyplot(plot_runtime(random_history, "Random Mutation - Generation Execution Time"))

        st.write("### Steps")
        st.pyplot(plot_steps(random_history, "Random Mutation - Steps per Generation"))

        st.write("### Operation Logs")
        st.text(random_logs if random_logs.strip() else "No logs available.")

    with tab3:
        st.write("### Final Best Solution")
        if llm_result["best_candidate"] is not None:
            st.code(llm_result["best_candidate"].solution, language="python")

        st.write("### Fitness Progression")
        st.pyplot(plot_fitness(llm_history, "LLM-Guided Mutation - Fitness Progression"))

        st.write("### Fitness Variance")
        st.pyplot(plot_variance(llm_history, "LLM-Guided Mutation - Fitness Variance"))

        st.write("### Runtime")
        st.pyplot(plot_runtime(llm_history, "LLM-Guided Mutation - Generation Execution Time"))

        st.write("### Steps")
        st.pyplot(plot_steps(llm_history, "LLM-Guided Mutation - Steps per Generation"))

        st.write("### Operation Logs")
        st.text(llm_logs if llm_logs.strip() else "No logs available.")
