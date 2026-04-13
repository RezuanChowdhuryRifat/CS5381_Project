import random
from typing import List, Any


# Step 1: Problem Setup & Initialization

class EvolutionConfig:
    """
    Configuration class to store the key parameters for the evolutionary loop.
    """
    def __init__(self):
        self.population_size: int = 5
        self.generations: int = 5
        self.mutation_rate: float = 0.3
        self.selection_strategy: str = "top-k"


class Initializer:
    """
    Handles the definition of initial algorithms and the generation of the first population.
    """
    def __init__(self, config: EvolutionConfig):
        self.config = config

    def get_base_algorithm(self, use_case: str) -> str:
        """
        Returns the foundational algorithm input for the specified use case.
        """
        if use_case == "cart_pole":
            return """def control_cart_pole(observation):
    pole_angle = observation[2]
    return 0 if pole_angle < 0 else 1"""

        elif use_case == "matrix_multiplication":
            return """def matrix_mult_3x3(A, B):
    C = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i][j] += A[i][k] * B[k][j]
    return C"""
        else:
            raise ValueError(f"Unsupported use case: {use_case}")

    def initialize_population(self, base_code: str, generator: Any, use_case: str) -> List[str]:
        """
        Generates the initial population with minor variations of the input algorithm.
        """
        population = [base_code]

        for _ in range(self.config.population_size - 1):
            mutated_code = generator.random_mutation(base_code, use_case)
            population.append(mutated_code)

        return population


# Step 2: Candidate Generation

class CandidateGenerator:
    """
    Responsible for generating candidate solutions via random mutation or LLM-guided refinement.
    """
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client

    def random_mutation(self, code: str, use_case: str) -> str:
        """
        Implements small random changes to the algorithms.
        """
        lines = code.strip().split('\n')

        if use_case == "cart_pole":
            threshold = round(random.uniform(-0.3, 0.3), 4)
            choice = random.random()

            if choice < 0.33:
                mutated_code = f"""def control_cart_pole(observation):
    pole_angle = observation[2]
    return 0 if pole_angle < {threshold} else 1"""
                return mutated_code

            elif choice < 0.66:
                factor = round(random.uniform(0.0, 0.5), 3)
                mutated_code = f"""def control_cart_pole(observation):
    pole_angle = observation[2]
    position = observation[0]
    return 0 if pole_angle + {factor} * position < {threshold} else 1"""
                return mutated_code

            else:
                factor = round(random.uniform(0.0, 0.5), 3)
                mutated_code = f"""def control_cart_pole(observation):
    pole_angle = observation[2]
    angular_velocity = observation[3]
    return 0 if pole_angle + {factor} * angular_velocity < {threshold} else 1"""
                return mutated_code

        elif use_case == "matrix_multiplication":
            if len(lines) > 4:
                idx1, idx2 = random.sample(range(2, len(lines)), 2)
                lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
            return '\n'.join(lines)

        return code

    def llm_guided_mutation(self, current_code: str, use_case: str, fitness_feedback: str) -> str:
        """
        Integrates a prompt-based LLM API to suggest improvements and refine candidate solutions.
        """
        if not self.llm_client:
            if use_case == "cart_pole":
                threshold = round(random.uniform(0.05, 0.18), 4)
                choice = random.random()

                if choice < 0.5:
                    return f"""def control_cart_pole(observation):
    pole_angle = observation[2]
    return 0 if pole_angle < {threshold} else 1"""
                else:
                    factor = round(random.uniform(0.05, 0.25), 3)
                    return f"""def control_cart_pole(observation):
    pole_angle = observation[2]
    angular_velocity = observation[3]
    return 0 if pole_angle + {factor} * angular_velocity < {threshold} else 1"""

            elif use_case == "matrix_multiplication":
                return current_code

            return current_code

        system_prompt = "You are an expert algorithm optimization agent."

        if use_case == "cart_pole":
            user_prompt = f"""
Task: Improve the Cart Pole control logic.
Goal: Maximize survival time, minimize average pole angle deviation, and minimize cart position drift.

Current Code:
{current_code}

Evaluation Feedback:
{fitness_feedback}

Instruction:
Return ONLY executable Python code for the function control_cart_pole(observation).
Do not include explanations, markdown, or backticks.
"""
        elif use_case == "matrix_multiplication":
            user_prompt = f"""
Task: Optimize 3x3 Matrix Multiplication.
Goal: Ensure correctness while minimizing operational cost (number of multiplications and additions).

Current Code:
{current_code}

Evaluation Feedback:
{fitness_feedback}

Instruction:
Return ONLY executable Python code for the function matrix_mult_3x3(A, B).
Do not include explanations, markdown, or backticks.
"""
        else:
            raise ValueError(f"Unsupported use case: {use_case}")

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )

            generated_code = response.choices[0].message.content
            generated_code = generated_code.replace("```python", "").replace("```", "").strip()

            if not generated_code:
                return current_code

            return generated_code

        except Exception as e:
            print(f"Error during LLM API call: {e}")
            return current_code