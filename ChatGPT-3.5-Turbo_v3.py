import os
from openai import OpenAI

class OpenAIClient:
    """OpenAI 客户端初始化与交互处理类。"""

    def __init__(self):
        """
        初始化 OpenAI 客户端。
            api_key (str): 用于认证的 API 密钥。
            base_url (str): API 的基本 URL。
        """
        api_key = os.getenv("OPENAI_API_KEY")
        base_url="https://api.chatanywhere.tech/v1"
        if not api_key:
            raise ValueError("API key for Groq is not set in the environment variables")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def ask_gpt(self, prompt):
        """
        利用 OpenAI GPT 模型生成回复。
        参数:
            prompt (str): 提示信息，作为模型的输入。
        返回:
            str: GPT 模型的回复内容。
        """
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI-controlled robotic arm. When given an object's name, determine the single part best for a safe grasp to hand it over to a human. Respond only with one noun that precisely describes that part. For example, for 'mug,' respond 'handle'; for 'book,' reply 'spine.' Ensure the grasp avoids damage to the object and risk to the person."},
                {"role": "user", "content": prompt},
            ]
        )
        return completion.choices[0].message.content


def main():
    print("Type 'bye' to end the conversation.")
    client = OpenAIClient()

    # 输入文件路径
    input_file_path = '/home/bigdata/PycharmProjects/pythonProject/objects_to_grasp.txt'
    # 输出文件路径
    output_file_path = '/home/bigdata/PycharmProjects/pythonProject/responsesGPT_v3.txt'

    count = 1
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            object_name = line.strip()  # 删除可能的空白字符
            if not object_name:
                continue  # 如果是空行则跳过
            response = client.ask_gpt(object_name)
            output_file.write(f"[{count}] Object: {object_name},        GPT Response: {response}\n")
            print(f"[{count}] GPT3.5:", response)
            count += 1


if __name__ == "__main__":
    main()
