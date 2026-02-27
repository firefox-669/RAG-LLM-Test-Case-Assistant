# -*- coding: utf-8 -*-
"""
LLM Handler - Simplified
"""

class SimpleLLM:
    def __init__(self):
        print("[OK] LLM demo mode initialized")

    def generate(self, prompt: str) -> str:
        if "test case" in prompt.lower() or "测试用例" in prompt:
            return """
Test Case ID: TC-001
Test Case Name: Functional Test
Test Type: Functional Test
Priority: High

Preconditions:
- System is running
- User is logged in

Test Steps:
1. Open the function page
2. Enter valid data
3. Click submit button
4. Verify the result

Expected Result:
- Operation successful
- Correct message displayed
- Data saved correctly

---

Test Case ID: TC-002
Test Case Name: Boundary Test
Test Type: Boundary Test
Priority: Medium

Preconditions:
- System is accessible

Test Steps:
1. Input minimum value
2. Input maximum value
3. Input boundary value
4. Observe system response

Expected Result:
- Boundary values handled correctly
- No exceptions
"""
        else:
            return """Based on your requirements, here are some testing suggestions:

1. Ensure test cases cover main functionality
2. Consider exception scenarios and boundary conditions
3. Write clear test steps
4. Define expected results clearly
5. Set reasonable priorities"""

class LLMHandler:
    def __init__(self):
        self.llm = SimpleLLM()
        print("[OK] LLM Handler initialized (demo mode)")

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            return f"Generation failed: {str(e)}"

    def generate_with_context(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        return self.generate(full_prompt)

_llm_handler = None

def get_llm_handler() -> LLMHandler:
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = LLMHandler()
    return _llm_handler
