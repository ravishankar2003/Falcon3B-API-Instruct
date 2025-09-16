# Falcon3B-API-Instruct

Fine-tuned Falcon3-3B-Instruct with PEFT-LoRA to produce consistently structured outputs (`domain`, `api_call`, `api_provider`, `explanation`, `code`) aligned to the dataset schema.  
Emphasis is on correctness of the **explanation** tag and semantic similarity of **code**, with notable gains in **ROUGE-L/BERTScore** for explanations and **CodeBERT similarity** for code.

## 📂 Access to Colab Notebooks

- [`instruct_tuning_2ndepoch.ipynb`](https://colab.research.google.com/drive/16M2li9oQJvfmKxb8QIYTd0rdc9qF8JPn?usp=sharing)  
- [`instruct_tuning_1stepoch.ipynb`](https://colab.research.google.com/drive/1rUhge5C3CT71O8HjoRBgTIBWhCmnkNYa?usp=sharing)  
- [`evaluation.ipynb`](https://colab.research.google.com/drive/1vSKdn636fn8_p5ZJukhOn8tfh3veLUwa?usp=sharing)  


## 🚀 Model & Dataset

**Base Model**: `tiiuae/Falcon3-3B-Instruct`  
**Fine-tuning**: LoRA (Low-Rank Adaptation)  
**Dataset**: Hugging Face Dataset from Gorilla / API Bench

### Output Format
The model generates structured responses with tagged components:

`<<<domain>>>: Natural Language Processing`

`<<<api_call>>>: AutoModel.from_pretrained()`

`<<<api_provider>>>: Hugging Face Transformers`

`<<<explanation>>>: Loads a pre-trained model from the Hugging Face Hub`

`<<<code>>>:
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
`



## 📈 Training Results

### Loss Progression
- **Training Loss**: 0.759 → 0.535 (29.6% reduction)
- **Validation Loss**: 0.766 → 0.545 (28.9% reduction)
- **Epochs**: 2 epochs
- **Training Approach**: Parameter-efficient LoRA fine-tuning

## 🎯 Evaluation Metrics

Evaluated on 911 test samples:

| Metric | Score | Description |
|--------|-------|-------------|
| **ROUGE-L** | 37.05% | Score for explanations|
| **BERTScore F1** | 90.56% | Semantic similarity for explanations |
| **CodeBERT Similarity** | 88.71% | Semantic code similarity |
| **Domain Accuracy** | 97.21% | Correct domain classification |
| **API Call Accuracy** | 85.59% | Correct API function identification |
| **API Provider Accuracy** | 97.32% | Correct provider identification |

## 🔧 Usage
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Auto-detect device and dtype for cross-device compatibility
if torch.cuda.is_available():
    device_dtype = torch.float16
    device_map = "auto"
else:
    device_dtype = torch.float32
    device_map = "cpu"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/Falcon3-3B-Instruct",
    dtype=device_dtype,  # Use torch_dtype instead of dtype
    device_map=device_map,
    trust_remote_code=True,
    low_cpu_mem_usage=True  # Better memory management
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model, 
    "./falcon3b_instruct_2ndepoch", 
    device_map=device_map
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-3B-Instruct")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Generate API code function
def generate_response(instruction):
    prompt = f"{instruction}\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Auto-move inputs to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=0,  # Disable to allow <<<>>> tags
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output.replace(prompt, "").strip()
    return response

# Test the model with a sample instruction
test_instruction = "Write code to load a pre-trained BERT model for text classification"
response = generate_response(test_instruction)
print(f"\n📝 Response:\n{response}")


```

## 📁 Files

- `evaluation.ipynb` - Evaluation pipeline
- `instruct_tuning_1stepoch.ipynb` - 1-epoch training
- `instruct_tuning_2ndepoch.ipynb` - 2-epoch training
- `falcon3b_instruct_2ndepoch/` - Final model weights

