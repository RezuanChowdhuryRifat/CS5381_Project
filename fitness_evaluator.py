import random
from statistics import mean, pstdev
from typing import Dict, Any, List

from evolution_engine import Candidate, EvaluationResult


def execute_code(code: str) -> Dict[str, Any]:
    local_env: Dict[str, Any] = {}
    exec(code, {}, local_env)
    return local_env


def _safe_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _matrix_multiply_reference(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    result = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            cell_sum = 0
            for k in range(3):
                cell_sum += a[i][k] * b[k][j]
            result[i][j] = cell_sum
    return result


def _to_3x3_list(result):
    """
    Convert candidate output into a plain 3x3 Python list when possible.
    Accepts nested lists, tuples, and numpy-like arrays via .tolist().
    """
    if hasattr(result, "tolist"):
        result = result.tolist()

    if not isinstance(result, (list, tuple)) or len(result) != 3:
        raise ValueError(f"Invalid matrix outer shape returned: {result}")

    normalized = []
    for row in result:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"Invalid matrix row returned: {row}")
        normalized.append([row[0], row[1], row[2]])

    return normalized


def _get_matrix_function(local_env: Dict[str, Any]):
    """
    Be flexible with function names because generated candidates may rename them.
    """
    possible_names = [
        "matrix_mult_3x3",
        "matrix_multiplication_3x3",
        "multiply_matrices_3x3",
        "multiply_matrices",
        "matrix_mult",
        "matmul_3x3",
        "matmul",
        "solve",
    ]

    for name in possible_names:
        fn = local_env.get(name)
        if callable(fn):
            return fn, name

    callable_items = [(name, value) for name, value in local_env.items() if callable(value)]
    if len(callable_items) == 1:
        return callable_items[0][1], callable_items[0][0]

    raise ValueError("No valid matrix multiplication function found")


def evaluate_cart_pole(candidate: Candidate) -> EvaluationResult:
    try:
        local_env = execute_code(candidate.solution)

        if "control_cart_pole" not in local_env:
            raise ValueError("Function control_cart_pole not found")

        control_fn = local_env["control_cart_pole"]

        test_cases = [
            (0.00, 0.00, 0.05, 0.00),
            (0.00, 0.00, -0.05, 0.00),
            (0.10, 0.00, 0.08, 0.02),
            (-0.10, 0.00, -0.08, -0.02),
            (0.00, 0.10, 0.03, 0.01),
            (0.00, -0.10, -0.03, -0.01),
            (0.20, 0.05, 0.10, 0.03),
            (-0.20, -0.05, -0.10, -0.03),
        ]

        max_steps_per_case = 100
        case_survival_times: List[int] = []
        total_angle = 0.0
        total_drift = 0.0
        total_steps = 0

        for initial_position, initial_velocity, initial_angle, initial_angular_velocity in test_cases:
            position = initial_position
            velocity = initial_velocity
            angle = initial_angle
            angular_velocity = initial_angular_velocity

            case_survival_time = 0

            for _ in range(max_steps_per_case):
                observation = [position, velocity, angle, angular_velocity]
                action = control_fn(observation)

                if action not in [0, 1]:
                    raise ValueError(f"Invalid action returned: {action}. Expected 0 or 1.")

                force = -1 if action == 0 else 1

                velocity += 0.1 * force
                position += velocity
                angular_velocity += 0.05 * force - 0.02 * angle
                angle += angular_velocity

                case_survival_time += 1
                total_steps += 1
                total_angle += abs(angle)
                total_drift += abs(position)

                if abs(angle) > 0.7 or abs(position) > 3.0:
                    break

            case_survival_times.append(case_survival_time)

        if total_steps == 0:
            raise ValueError("Total evaluated steps is zero; cannot compute fitness.")

        total_possible_survival = len(test_cases) * max_steps_per_case
        survival_time = sum(case_survival_times)
        avg_survival = mean(case_survival_times)
        std_survival = pstdev(case_survival_times) if len(case_survival_times) > 1 else 0.0

        avg_angle = total_angle / total_steps
        avg_drift = total_drift / total_steps

        normalized_survival = survival_time / total_possible_survival
        stability_score = 1.0 - _safe_clip(std_survival / max_steps_per_case)

        angle_score = 1.0 - _safe_clip(avg_angle / 0.7)
        drift_score = 1.0 - _safe_clip(avg_drift / 3.0)
        control_quality = 0.6 * angle_score + 0.4 * drift_score

        fitness = (
            0.70 * normalized_survival
            + 0.20 * stability_score
            + 0.10 * control_quality
        ) * 100.0

        return EvaluationResult(
            candidate_id=candidate.id,
            fitness=fitness,
            metrics={
                "survival_time": survival_time,
                "avg_survival": avg_survival,
                "std_survival": std_survival,
                "normalized_survival": normalized_survival,
                "stability_score": stability_score,
                "avg_angle": avg_angle,
                "avg_drift": avg_drift,
                "angle_score": angle_score,
                "drift_score": drift_score,
                "control_quality": control_quality,
                "steps": total_steps,
                "num_test_cases": len(test_cases),
                "max_steps_per_case": max_steps_per_case,
            },
            passed=True,
        )

    except Exception as e:
        return EvaluationResult(
            candidate_id=candidate.id,
            fitness=float("-inf"),
            metrics={"debug_error": str(e)},
            passed=False,
            error=str(e),
        )


def evaluate_matrix(candidate: Candidate) -> EvaluationResult:
    try:
        local_env = execute_code(candidate.solution)
        func, detected_name = _get_matrix_function(local_env)

        rng = random.Random(42)
        test_cases = [
            (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            )
        ]

        for _ in range(9):
            a = [[rng.randint(-5, 9) for _ in range(3)] for _ in range(3)]
            b = [[rng.randint(-5, 9) for _ in range(3)] for _ in range(3)]
            test_cases.append((a, b))

        total_correct_cells = 0
        total_cells = 0
        passed_cases = 0

        for a, b in test_cases:
            raw_result = func(a, b)
            result = _to_3x3_list(raw_result)
            expected = _matrix_multiply_reference(a, b)

            correct_cells = 0
            for i in range(3):
                for j in range(3):
                    total_cells += 1
                    if result[i][j] == expected[i][j]:
                        correct_cells += 1
                        total_correct_cells += 1

            if correct_cells == 9:
                passed_cases += 1

        correctness_score = total_correct_cells / total_cells if total_cells else 0.0
        passed_case_score = passed_cases / len(test_cases) if test_cases else 0.0

        multiplications = candidate.solution.count("*")
        additions = candidate.solution.count("+")
        operations = multiplications + additions

        expected_naive_operations = 45
        efficiency_score = 1.0 - _safe_clip(operations / (expected_naive_operations * 2))
        efficiency_score = max(efficiency_score, 0.0)

        fitness = (
            0.80 * correctness_score
            + 0.15 * passed_case_score
            + 0.05 * efficiency_score
        ) * 100.0

        return EvaluationResult(
            candidate_id=candidate.id,
            fitness=fitness,
            metrics={
                "detected_function_name": detected_name,
                "correctness_score": correctness_score,
                "passed_case_score": passed_case_score,
                "passed_cases": passed_cases,
                "total_cases": len(test_cases),
                "correct_cells": total_correct_cells,
                "total_cells": total_cells,
                "multiplications": multiplications,
                "additions": additions,
                "operations": operations,
                "efficiency_score": efficiency_score,
                "steps": operations,
            },
            passed=True,
        )

    except Exception as e:
        return EvaluationResult(
            candidate_id=candidate.id,
            fitness=float("-inf"),
            metrics={"debug_error": str(e)},
            passed=False,
            error=str(e),
        )


def fitness_evaluator(candidate: Candidate, use_case: str) -> EvaluationResult:
    if use_case == "cart_pole":
        return evaluate_cart_pole(candidate)
    elif use_case == "matrix_multiplication":
        return evaluate_matrix(candidate)
    else:
        return EvaluationResult(
            candidate_id=candidate.id,
            fitness=float("-inf"),
            metrics={},
            passed=False,
            error=f"Unknown use case: {use_case}",
        )
