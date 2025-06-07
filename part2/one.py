import os
from camel.agents import ChatAgent
from camel.configs import ZhipuAIConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType


from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('QWEN_API_KEY')
print(api_key)
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key= '9895cbb2-44ce-4496-8514-5ccd6100cc49'


)

task_kwargs = {
    'task_prompt': '写一本关于AI社会的未来的书。',
    'with_task_specify': True,
    'task_specify_agent_kwargs': {'model': model}
}

user_role_kwargs = {
    'user_role_name': 'AI专家',
    'user_agent_kwargs': {'model': model}
}

assistant_role_kwargs = {
    'assistant_role_name': '对AI感兴趣的作家',
    'assistant_agent_kwargs': {'model': model}
}

society = RolePlaying(
    **task_kwargs,             # 任务参数
    **user_role_kwargs,        # 指令发送者的参数
    **assistant_role_kwargs,   # 指令接收者的参数
    # with_critic_in_the_loop=True,
    output_language='中文'
)


def is_terminated(response):
    if response.terminated :
        role = response.msg.role_type.name
        reason = response.info['termination_reasons']
        print(f'AI{role} 因为{reason} 而终止')

def run(society, round_limit: int=10):

    # 获取AI助手到AI用户的初始消息
    input_msg = society.init_chat()

    # 开始互动会话
    for _ in range(round_limit):

        # 获取这一轮的两个响应
        assistant_response, user_response = society.step(input_msg)

        # 检查终止条件
        if is_terminated(assistant_response) or is_terminated(user_response):
            break

        # 获取结果
        print(f'[AI 用户] {user_response.msg.content}.\n')
        # 检查任务是否结束
        if 'CAMEL_TASK_DONE' in user_response.msg.content:
            break
        print(f'[AI 助手] {assistant_response.msg.content}.\n')

        # 获取下一轮的输入消息
        input_msg = assistant_response.msg

    return None

run(society)