import dashscope
from http import HTTPStatus
dashscope.api_key="sk-"

def get_completion(prompt, model=dashscope.Generation.Models.qwen_turbo, temperature=0): 
    messages = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # set the result to be "message" format.
    )
    if response.status_code == HTTPStatus.OK:
        return response
    else:
        print(response) 


prompt = f"""
Translate the following text to 上海话 和 四川话 in both the \
formal and informal forms: 
'Would you like to order a pillow?'
"""

response = get_completion(prompt)
print(response.output.choices[0].message["content"])