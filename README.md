环境配置  requirements.txt  
运行代码  annotate_file_multi-shot.py  
few-shot提示文件 Prompt  
运行脚本  run.sh  
在run.sh中 配置 ckpt_dir 参数为llama模型文件  
tokenizer_path 参数为llama分词文件 
替换该参数为不同llama模型  7b  13b  70b 
模型文件路径为 model-7b   model-13b  model-70b 需要从huggingface授权下载  
参数举例（model-7b） --ckpt_dir ./model-7b   --tokenizer  ./model-7b/tokenizer.model  
运行时出现端口占用，需要更改 --master_port  端口参数
