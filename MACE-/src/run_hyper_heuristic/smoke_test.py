from openai import OpenAI
import os
import traceback
from typing import Tuple, Optional
import re
from src.run_hyper_heuristic.run_hyper_heuristic import run_hyper_heuristic
from src.run_hyper_heuristic.helper_function import extract_code_from_response, save_generated_heuristic


class SmokeTestRunner:

    def __init__(
        self,
        problem: str = None,
        max_fix_attempts: int = 3,
        api_key: str = None
    ):
        self.problem = problem
        self.max_fix_attempts = max_fix_attempts
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

    def _clean_code_aggressive(self, code: str) -> str:
        original_code = code

        code = re.sub(r'\*\*\*python_code:\s*\n?', '', code)
        code = re.sub(r'\n?\*\*\*\s*$', '', code)
        code = re.sub(r'^\*\*\*\s*\n?', '', code)

        code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'```$', '', code)

        if '***' in code or '```' in code:
            max_iterations = 5
            for _ in range(max_iterations):
                old_code = code
                code = re.sub(r'\*\*\*python_code:\s*\n?', '', code)
                code = re.sub(r'\n?\*\*\*\s*', '', code)
                code = re.sub(r'```python\s*\n?', '', code)
                code = re.sub(r'```\s*', '', code)
                if code == old_code:
                    break

        code = code.strip()

        if '```' in code or '***' in code:
            print("Warning: markers still present after cleaning")
            print(f"Original length: {len(original_code)}, cleaned length: {len(code)}")
            print(f"First 200 characters after cleaning:\n{code[:200]}")

        return code

    def run_smoke_test(self, heuristic_file) -> Tuple[bool, Optional[str]]:
        try:
            heuristic_name = os.path.splitext(os.path.basename(heuristic_file))[0]

            print(f"\n{'='*80}")
            print(f"Running Smoke Test: {heuristic_name}")
            print(f"Problem type: {self.problem}")
            print(f"{'='*80}")

            with open(heuristic_file, 'r', encoding='utf-8') as f:
                code = f.read()

            needs_cleaning = False
            if '```' in code:
                print("Detected Markdown marker (```)")
                needs_cleaning = True
            if '***' in code:
                print("Detected custom marker (***)")
                needs_cleaning = True

            if needs_cleaning:
                print("Cleaning code...")
                cleaned_code = self._clean_code_aggressive(code)
                with open(heuristic_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_code)
                print("Code cleaned and saved")

            problem = self.problem
            test_data = "smoke_data"
            llm_config_file = os.path.join("output", "llm_config", "azure_gpt_4o.json")
            heuristic_dir = "basic_heuristics"
            iterations_scale_factor = 1.0
            result_dir = "result"

            validation_result = run_hyper_heuristic(
                problem=problem,
                heuristic=heuristic_name,
                llm_config_file=llm_config_file,
                heuristic_dir=heuristic_dir,
                test_data=test_data,
                iterations_scale_factor=iterations_scale_factor,
                result_dir=result_dir
            )

            validation_symbol = True
            for i in validation_result:
                try:
                    if i > 0:
                        pass
                except Exception:
                    validation_symbol = False

            if validation_symbol:
                print("Smoke Test passed")
                return True, None
            else:
                error_msg = "Validation failed: invalid values in results"
                print(f"Smoke Test failed:\n{error_msg}")
                return False, error_msg

        except SyntaxError as e:
            error_msg = f"SyntaxError: {str(e)}"
            print(f"Smoke Test failed:\n{error_msg}")
            print(f"\nTraceback:\n{traceback.format_exc()}")
            return False, error_msg

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Smoke Test failed:\n{error_msg}")
            print(f"\nTraceback:\n{traceback.format_exc()}")
            return False, error_msg

    def fix_code_with_llm(
        self,
        original_code: str,
        error_message: str,
        attempt: int
    ) -> Optional[str]:
        print(f"\n{'='*80}")
        print(f"Fix attempt {attempt}/{self.max_fix_attempts}")
        print(f"{'='*80}")

        fix_prompt = f"""You are an expert Python debugger. Fix this code.

Original Code:
{original_code}

Error:
{error_message}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY executable Python code
2. No markdown formatting
3. No explanations
4. Start directly with Python code
5. The response must be immediately executable

Fixed code:"""

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )

            print("Calling LLM to fix code...")
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python code fixer. Respond only with executable Python code."
                    },
                    {
                        "role": "user",
                        "content": fix_prompt
                    }
                ],
                temperature=0.2
            )

            llm_response = response.choices[0].message.content

            fixed_code = extract_code_from_response(llm_response)
            if not fixed_code:
                fixed_code = self._clean_code_aggressive(llm_response)
            else:
                fixed_code = self._clean_code_aggressive(fixed_code)

            if fixed_code:
                if '```' in fixed_code or '***' in fixed_code:
                    fixed_code = self._clean_code_aggressive(fixed_code)

                print("LLM generated fixed code")
                return fixed_code
            else:
                print("Failed to extract code from LLM response")
                print(f"First 200 characters of response:\n{llm_response[:200]}")
                return None

        except Exception as e:
            print(f"LLM fix failed: {str(e)}")
            traceback.print_exc()
            return None

    def test_and_fix_cycle(
        self,
        heuristic_file: str,
        save_fixed: bool = True
    ) -> Tuple[bool, str]:
        current_file = heuristic_file

        success, error_msg = self.run_smoke_test(current_file)
        if success:
            return True, current_file

        for attempt in range(1, self.max_fix_attempts + 1):
            with open(current_file, 'r', encoding='utf-8') as f:
                current_code = f.read()

            current_code = self._clean_code_aggressive(current_code)
            fixed_code = self.fix_code_with_llm(current_code, error_msg, attempt)

            if not fixed_code:
                print("Fix failed, aborting")
                return False, current_file

            if save_fixed:
                fixed_file = save_generated_heuristic(
                    fixed_code,
                    output_dir=os.path.dirname(current_file)
                )
                print(f"Code saved to: {fixed_file}")
                current_file = fixed_file
            else:
                with open(current_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_code)
                print(f"Code saved to: {current_file}")

            success, error_msg = self.run_smoke_test(current_file)
            if success:
                print(f"\n{'='*80}")
                print(f"Fix succeeded after {attempt} attempts")
                print(f"{'='*80}")
                return True, current_file

        print(f"\n{'='*80}")
        print(f"Aborted after {self.max_fix_attempts} failed attempts")
        print(f"{'='*80}")
        return False, current_file


def standalone_smoke_test(
    heuristic_file: str,
    problem: str = None,
    api_key: str = None
) -> bool:
    """
    Standalone smoke test function.

    Args:
        heuristic_file: Path to heuristic file
        problem: Problem type (tsp, jssp, cvrp, psp)
        api_key: API key

    Returns:
        Whether the test passed
    """
    tester = SmokeTestRunner(
        problem=problem,
        max_fix_attempts=3,
        api_key=api_key
    )

    success, _ = tester.test_and_fix_cycle(heuristic_file)
    return success
