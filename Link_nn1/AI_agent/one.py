from openai import OpenAI
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='aae43e29-dbbf-4fa3-9301-12941848f807', # ModelScope Token
)

response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', # ModelScope Model-Id
    messages=[
        {
            'role': 'user',
            'content': '你好'
        }
    ],
    stream=True
)
done_reasoning = False




for chunk in response:
    reasoning_chunk = chunk.choices[0].delta.reasoning_content
    answer_chunk = chunk.choices[0].delta.content
    if reasoning_chunk != '':
        print(reasoning_chunk, end='',flush=True)
    elif answer_chunk != '':
        if not done_reasoning:
            print('\n\n === Final Answer ===\n')
            done_reasoning = True
        print(answer_chunk, end='',flush=True)

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    url='https://api-inference.modelscope.cn/v1/',
    api_key='aae43e29-dbbf-4fa3-9301-12941848f80'
)

agent = ChatAgent(
    model=model,
    output_language='中文'
)

response = agent.step("你好，你是谁？")
print(response.msgs[0].content)





