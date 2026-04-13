CS5381 Evolutionary Algorithm Project (Group # 8)

 Overview

This project implements an Evolutionary Algorithm framework to compare three approaches:
 No Evolution
 Random Mutation
 LLM-Guided Mutation

 REQUIRED ENVIRONMENTS
•	Python 3.9+
•	Operating System: Windows / MacOS

LIBRARIES USED
•	Numpy
•	Matplotlib
•	Pandas
•	Openai 

FLOW OF EXECUTION
•	Initialize System
	Load Initial Algorithm
	Set Parameters (population size, generations, mutation rate)
•	Candidate Generation
	Apply:
	Random Mutation
	LLM-guided Mutation
	No Evolution
•	Evalaution
•	Fitness Calculation
	Compute weighted fitness score
•	Selection
	Retain top-k candidates
•	Iteration
	Repeat mutation → evaluation → selection for N generations
•	Output
	Best Evolved Solution
	Fitness Progression
	Comparative Results

The goal is to analyze how different mutation strategies affect fitness improvement, convergence, and performance.
 Features
 Weighted Fitness Evaluation (CartPole)
Streamlit UI for interactive visualization

Performance Tracking
   Best Fitness
   Average Fitness
   Variance
   Runtime
   Steps per Generation
CSV Export for all experiment modes
Comparison Analysis across strategies

How to Run

 1. Install Dependencies
pip install streamlit matplotlib openai

 2. Navigate to Project Folder
cd path_to_project

 3. Add API Key (for LLM mode)
Open ui_app.py and set:
llm_client = OpenAI(api_key="YOUR_API_KEY")

 4. Run the Application
streamlit run ui_app.py

Usage
 Select CartPol use case
 Click Run Experiments
 
 View:
   Fitness progression
   Variance
   Runtime
   Steps per generation
   Final comparison across all methods

 Output
CSV files are automatically generated:
_no_evolution.csv
_random_mutation.csv
_llm_mutation.csv

Important Note
LLM-guided mutation requires an API key.
The project can still be evaluated using:
 Provided CSV results
 Final report with graphs and analysis

Results Summary
No Evolution: Constant performance
Random Mutation: Minor improvements
LLM-Guided Mutation: Significant fitness improvement and faster convergence

Report
The project includes a detailed report with:
 Graphs for all metrics
 Comparative analysis
 Observations and conclusions

 Contribution
Adil Shah
Implemented:
 Fitness Evaluation
 UI Visualization (Streamlit)
 Performance Tracking & Metrics
 CSV Export
 Comparison Analysis
 Final Report

Beatrice implemented the selection.py, wrote the slides and compile documents for submission


Conclusion
LLM-guided mutation significantly outperforms traditional approaches by achieving higher fitness and better convergence, demonstrating the effectiveness of intelligent mutation strategies in evolutionary algorithms.
