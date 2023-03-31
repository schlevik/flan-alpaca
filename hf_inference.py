import fire
import torch
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel


def test(tokenizer, model, prompt, temperature=0.1, top_p=0.7, top_k=40, num_beams=4, max_new_tokens=160, **kwargs):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generation_config = GenerationConfig(
        temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, **kwargs
    )
    generation_output = model.generate(
        input_ids=input_ids.cuda(),
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )

    print(tokenizer.batch_decode(generation_output.sequences)[0])


def infer(
    base_model: str = "google/flan-t5-xxl",
    adapter_path: str = "results-xxl/",
    prompt: str = "Instruction:\nParaphrase the input phrase as if it was spoken in the Victorian Era!\nInput: I'm gonna waste that Punk!",
    temperature: float = 0.1,
    top_p: float = 0.7,
    top_k: int = 40,
    num_beams: int = 4,
    max_new_tokens: int = 160,
    load_in_8bit: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, load_in_8bit=load_in_8bit, torch_dtype=torch.float16, device_map="auto"
    )
    model = (
        PeftModel.from_pretrained(
            model,
            adapter_path,
            torch_dtype=torch.float16,
        )
        .float()
        .eval()
    )
    test(tokenizer, model, prompt, temperature, top_p, top_k, num_beams, max_new_tokens)


if __name__ == "__main__":
    fire.Fire(infer)
