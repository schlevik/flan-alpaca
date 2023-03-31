import shutil
from pathlib import Path

import torch
from fire import Fire
from huggingface_hub import HfApi
from lightning_fabric import seed_everything

from training import LightningModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


def test_model(
    base_model: str,
    adapter_weights: str,
    prompt: str = "",
    load_in_8bit: bool = False,
    device: str = "cuda",
    temperature: float = 1.0,
    top_p: float = 0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens: int = 128,
):
    if not prompt:
        prompt = "Write a short email to show that 42 is the optimal seed for training neural networks"

    # model: LightningModel = LightningModel.load_from_checkpoint(path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # seed_everything(model.hparams.seed)
    """ with torch.inference_mode():
        model.model.eval()
        model = model.to(device)
        input_ids = input_ids.to(device)
        outputs = model.model.generate(input_ids, max_length=max_length, do_sample=True) """
    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            adapter_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, device_map={"": device}, low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(
            model,
            adapter_weights,
            device_map={"": device},
        )

    if not load_in_8bit:
        model.half()

    model.eval()

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    generation_output = model.generate(
                input_ids=input_ids.to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
    print(tokenizer.batch_decode(generation_output.sequences)[0])

    """
    Example output (outputs/model/base/epoch=2-step=2436.ckpt):
    <pad> Dear [Company Name], I am writing to demonstrate the feasibility of using 42 as an optimal seed
    for training neural networks. I am sure that this seed will be an invaluable asset for the training of 
    these neural networks, so let me know what you think.</s>
    """


def export_to_hub(path: str, repo: str, temp: str = "temp"):
    if Path(temp).exists():
        shutil.rmtree(temp)

    model = LightningModel.load_from_checkpoint(path)
    model.model.save_pretrained(temp)
    model.tokenizer.save_pretrained(temp)
    del model  # Save memory?

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=temp)


"""
huggingface-cli login

p inference.py export_to_hub \
--path "outputs_unclean/model/xl/epoch=2-step=2439.ckpt" \
--repo declare-lab/flan-alpaca-xl

p inference.py export_to_hub \
--path "outputs/model/xxl/epoch=0-step=203.ckpt" \
--repo declare-lab/flan-alpaca-xxl

"""


if __name__ == "__main__":
    Fire()
