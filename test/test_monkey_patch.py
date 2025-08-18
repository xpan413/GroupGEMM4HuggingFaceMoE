from groupgemm4huggingface_moe import AutoGroupGEMMForCausalLM
from transformers import AutoConfig
import logging
import torch

logging.basicConfig(level=logging.INFO)

# print(transformers.AutoModelForCausalLM)
if __name__ == "__main__":
    config = AutoConfig.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")
    model = AutoGroupGEMMForCausalLM.from_config(config)

    device = "cuda"
    batch_size = 4
    seq_length = 256
    model.to(device)

    vocab_size = config.vocab_size
    # 随机生成input_ids，范围为[0, vocab_size-1]
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_length), device=device, dtype=torch.long
    )

    # attention_mask通常全为1，表示所有token都参与计算
    attention_mask = torch.ones(
        (batch_size, seq_length), device=device, dtype=torch.long
    )

    # labels用于计算损失，通常与input_ids相同
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    loss.backward()
