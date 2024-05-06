import os
from dotenv import load_dotenv
from groq import Groq

# 加载.env文件
load_dotenv("/home/bigdata/PycharmProjects/pythonProject/GROQ_API_KEY.env")

class GroqClient:
    """Groq API 客户端初始化与交互处理类。"""

    def __init__(self):
        """初始化 Groq 客户端，使用环境变量中的 API 密钥。"""
        api_key = os.getenv("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1"
        if not api_key:
            raise ValueError("API key for Groq is not set in the environment variables")
        self.client = Groq(api_key=api_key)

    def ask_groq(self, prompt):
        """
        利用 Groq API 模型生成回复。
        参数:
            prompt (str): 提示信息，作为模型的输入。
        返回:
            str: Groq 模型的回复内容。
        """
        completion = self.client.chat.completions.create(
            model="Llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an AI-controlled robotic hand. Please respond with just one noun that best describes the safest part of an object to grasp and don't reply the explaination. This part should ensure a secure handover without damaging the object or posing risk to a person. For example, for 'mug,' reply 'handle'; for 'book,' reply 'spine'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=2048,
            top_p=1,
            stop="bye",
            stream=False,
        )
        return completion.choices[0].message.content

def main():
    print("Type 'bye' to end the conversation.")
    client = GroqClient()

    # 输入文件路径
    input_file_path = '/home/bigdata/PycharmProjects/pythonProject/objects_to_grasp.txt'
    # 输出文件路径
    output_file_path = '/home/bigdata/PycharmProjects/pythonProject/responses_Llama3_v3.txt'

    count = 1
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            object_name = line.strip()  # 删除可能的空白字符
            if not object_name:
                continue  # 如果是空行则跳过
            response = client.ask_groq(object_name)
            output_file.write(f"[{count}] Object: {object_name},        Llama3 Response: {response}\n")
            print(f"[{count}] Llama3:", response)
            count += 1

if __name__ == "__main__":
    main()