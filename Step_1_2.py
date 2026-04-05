import random
import re
from typing import List, Any


# Step 1: Problem Setup & Initialization

class EvolutionConfig:
    """
    Configuration class to store the key parameters for the evolutionary loop.
    """
    def __init__(self):
        # Configure key parameters as outlined in the project requirements
        self.population_size: int = 10
        self.generations: int = 10
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
            # Base logic: Push left (0) if the pole tilts left (theta < 0), else push right (1)
            return """def control_cart_pole(observation):
    # observation[2] represents the pole angle
    pole_angle = observation[2]
    return 0 if pole_angle < 0 else 1"""
        
        elif use_case == "matrix_multiplication":
            # Base logic: Standard O(n^3) nested loops for 3x3 matrix multiplication
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
        # Retain one pristine copy of the original base code
        population = [base_code] 
        
        # Fill the rest of the population using random mutations
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
        Implements small random changes to the algorithms (parameter tweaks or line swaps).
        """
        lines = code.strip().split('\n')
        
        if use_case == "cart_pole":
            # Strategy: Parameter Tweaks
            # Multiply any numeric value in the code by a random factor between 0.8 and 1.2
            def tweak_number(match):
                value = float(match.group())
                tweaked_value = value * random.uniform(0.8, 1.2)
                return str(round(tweaked_value, 4))
            
            # Regex to find integers and floating-point numbers
            mutated_code = re.sub(r"[-+]?\d*\.\d+|\d+", tweak_number, code)
            return mutated_code

        elif use_case == "matrix_multiplication":
            # Strategy: Line Swaps
            # Randomly swap two lines within the loop body to alter execution order
            if len(lines) > 4:
                # Pick two random line indices, skipping the function definition and initialization
                idx1, idx2 = random.sample(range(2, len(lines)), 2)
                lines[idx1], lines[idx2] = lines[idx2], lines[idx1]
            return '\n'.join(lines)
            
        return code

    def llm_guided_mutation(self, current_code: str, use_case: str, fitness_feedback: str) -> str:
        """
        Integrates a prompt-based LLM API to suggest improvements and refine candidate solutions.
        """
        if not self.llm_client:
            # Return the original code if no LLM client is configured
            return current_code 
            
        system_prompt = "You are an expert algorithm optimization agent."
        
        # Prompt
        if use_case == "cart_pole":
            user_prompt = f"""
            Task: Improve the Cart Pole control logic.
            Goal: Maximize survival time, minimize average pole angle deviation, and minimize cart position drift.
            
            Current Code: 
            {current_code}
            
            Evaluation Feedback: {fitness_feedback}
            
            Instruction: Refine the code to address the feedback. Return ONLY the executable Python code.
            """
        elif use_case == "matrix_multiplication":
            user_prompt = f"""
            Task: Optimize 3x3 Matrix Multiplication.
            Goal: Ensure correctness while minimizing operational cost (number of multiplications and additions).
            
            Current Code: 
            {current_code}
            
            Evaluation Feedback: {fitness_feedback}
            
            Instruction: Suggest a more efficient algorithmic sequence. Return ONLY the executable Python code.
            """
        else:
            raise ValueError(f"Unsupported use case: {use_case}")

        try:
            # Use LLM model API
            response = self.llm_client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            # Strip any markdown formatting the LLM might output
            return response.choices[0].message.content.strip().replace("```python", "").replace("```", "")
            
        except Exception as e:
            print(f"Error during LLM API call: {e}")
            return current_code