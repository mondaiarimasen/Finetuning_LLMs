# Fine-tuning GPT-2 for Custom Text Generation

This repository contains code for fine-tuning the GPT-2 model for text generation tasks, with a flexible design that can be easily adapted to fine-tune other language models, such as LLaMA2. The repository supports text data preprocessing, model training, and evaluation, leveraging the Hugging Face `transformers` library for seamless integration with a variety of models.

## Features
- Fine-tuning GPT-2 on custom datasets.
- Easy adaptation to other models like LLaMA2 by changing minimal configurations.
- Text data preprocessing (tokenization, padding, etc.).
- Support for GPU/TPU acceleration.
- Generation of custom text based on trained models.

## Requirements

Ensure that you have the following dependencies installed:

```bash
pip install torch transformers datasets accelerate
```

## Customization

Feel free to modify the scripts for different configurations, hyperparameters, or datasets. The architecture supports custom batch sizes, learning rates, and other training parameters via command-line arguments.

As an example, the current code can be used to finetune GPT-2 on the wikitext103 dataset, achieving a perplexity of 19. Please find both the training and validation code in the repo. 

## References

This project uses the Hugging Face `transformers` library, which provides the infrastructure for loading, fine-tuning, and deploying pre-trained models:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
