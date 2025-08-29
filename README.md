# Ramiz Rovshen GPT-2 Poetry Generator

A fine-tuned GPT-2 model for generating Azerbaijani poetry in the style of Ramiz Rovshen, a renowned Azerbaijani poet.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bahramzada/ramizrovshen-gpt2-poetry/blob/main/Ramiz_Rovshan_Poetry_gpt2.ipynb)

## Overview

This project fine-tunes OpenAI's GPT-2 model on a curated collection of Azerbaijani poems by Ramiz Rovshen to create an AI system capable of generating new poetry in his distinctive style. The model learns the linguistic patterns, metaphorical structures, and thematic elements characteristic of Rovshen's work.

## About Ramiz Rovshen

Ramiz Rovshen (1946-2022) was an influential Azerbaijani poet known for his profound and emotionally resonant verse. His poetry explores themes of love, nature, existential reflection, and the human condition, written in a distinctly modern Azerbaijani literary style.

## Features

- 🎭 **Poetry Generation**: Generate original Azerbaijani poems in Ramiz Rovshen's style
- 🧠 **GPT-2 Based**: Built on OpenAI's powerful GPT-2 language model
- 📚 **Extensive Training**: Trained on a comprehensive collection of cleaned poems (7,691 lines)
- 🔧 **Customizable**: Adjustable generation parameters for different poetic outputs
- 📊 **Experiment Tracking**: Integrated with Weights & Biases for monitoring training progress

## Dataset

The training dataset consists of:
- **Source**: Cleaned poems by Ramiz Rovshen
- **Size**: 7,691 lines of poetry
- **Language**: Azerbaijani
- **Format**: UTF-8 encoded text file
- **Preprocessing**: Poems are separated and cleaned for optimal training

## Model Details

- **Base Model**: GPT-2 (OpenAI)
- **Training Epochs**: 100
- **Batch Size**: 2 per device
- **Max Sequence Length**: 128 tokens
- **Training Loss**: 0.3788 (final)
- **Checkpoints**: Saved every 500 steps
- **Total Training Steps**: 16,300

## Requirements

```
torch
transformers
datasets
wandb
numpy
jupyter
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bahramzada/ramizrovshen-gpt2-poetry.git
cd ramizrovshen-gpt2-poetry
```

2. Install dependencies:
```bash
pip install torch transformers datasets wandb numpy jupyter
```

3. Set up Weights & Biases (optional, for training monitoring):
```bash
wandb login
```

## Usage

### Using Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Run all cells in the notebook
3. The notebook will:
   - Load and preprocess the poem dataset
   - Fine-tune GPT-2 on Ramiz Rovshen's poetry
   - Generate new poems using the trained model

### Local Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook Ramiz_Rovshan_Poetry_gpt2.ipynb
```

2. Follow the step-by-step process in the notebook

### Quick Generation Example

```python
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model (after training)
model_path = "./gpt2-poetry-ramizrovshen/checkpoint-16000"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate poetry
prompt = "Qadınım"  # "My woman" in Azerbaijani
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
```

## Training Process

The training process includes:

1. **Data Loading**: Load poems from `cleaned_poems.txt`
2. **Dataset Creation**: Convert text to Hugging Face Dataset format
3. **Tokenization**: Process text using GPT-2 tokenizer
4. **Model Training**: Fine-tune GPT-2 with custom training arguments
5. **Checkpointing**: Save model states for inference

### Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./gpt2-poetry-ramizrovshen",
    per_device_train_batch_size=2,
    num_train_epochs=100,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
)
```

## File Structure

```
ramizrovshen-gpt2-poetry/
├── README.md                           # This file
├── Ramiz_Rovshan_Poetry_gpt2.ipynb    # Main training and generation notebook
├── cleaned_poems.txt                   # Training dataset (Ramiz Rovshen's poems)
└── .git/                              # Git repository data
```

## Results

The fine-tuned model successfully:
- Generates coherent Azerbaijani poetry
- Maintains thematic consistency with Rovshen's style
- Produces grammatically correct Azerbaijani text
- Captures poetic meter and rhythm patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source. Please note that the poems by Ramiz Rovshen may be subject to copyright restrictions.

## Acknowledgments

- **Ramiz Rovshen**: For his beautiful poetry that serves as the foundation for this project
- **OpenAI**: For the GPT-2 model architecture
- **Hugging Face**: For the transformers library and model hosting
- **Weights & Biases**: For experiment tracking and monitoring

## Citation

If you use this model or dataset in your research, please cite:

```bibtex
@misc{ramizrovshen-gpt2-poetry,
  title={Ramiz Rovshen GPT-2 Poetry Generator},
  author={bahramzada},
  year={2024},
  url={https://github.com/bahramzada/ramizrovshen-gpt2-poetry}
}
```

---

*"Poetry is the language of the soul, and AI helps us explore new dimensions of creative expression."*