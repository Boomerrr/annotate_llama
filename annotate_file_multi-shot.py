# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = 20,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    valid_file_path = "./prompt_valid.txt"
    with open(valid_file_path, "r", encoding="utf-8") as file:
        prompt_valid_data = file.readlines()

    for num in range(1, 21):
        # 读取prompt文件
        input_file_path = f"./Prompt/promtp_{num}-shot.txt"
        with open(input_file_path, "r", encoding="utf-8") as input_file:
            requirement_prompt = input_file.read()

        # 根据prompt文件组件多个输入
        prompts_input: List[str] = []
        for line in prompt_valid_data:
            content,review = line.strip().split('\t')
            prompts_input.append(requirement_prompt +"\n"+ content + "->")
        # print(len(prompts_input))

        # 传入模型进行输出
        chunk_size = 4
        for i in range(0,len(prompts_input),chunk_size):
            chunk_prompts = prompts_input[i:i+chunk_size]
            results = generator.text_completion(
                        chunk_prompts,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
            )

            # 展示模型输出结果
            for prompt, result in zip(chunk_prompts, results):
                result = result['generation'].split('\n')
                result = result[0]
                print("\n==================================\n")
                print(f" {result}")
                print("\n==================================\n")

            # 将模型输出结果写入文件
            output_file_path = f"./prompt_output/prompt_output_{num}-shot_{ckpt_dir}.txt"

            with open(output_file_path,"a",encoding="utf-8") as file:
                for data_line,result in zip(prompt_valid_data[i:i+chunk_size],results):
                    result = result['generation'].split('\n')
                    result = result[0]
                    content,review = data_line.strip().split('\t')
                    file.write(f"{content}\t{review}\t{result}\n")

if __name__ == "__main__":
    fire.Fire(main)
