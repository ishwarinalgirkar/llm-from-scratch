# llm-from-scratch

A component-by-component implementation of every building block in a 
modern large language model, written from first principles in pure Python 
and PyTorch - no HuggingFace, no black boxes.

## Purpose

This repo exists for one reason: to build the kind of deep understanding 
that holds up under pressure. Every module is implemented from scratch, 
heavily commented, and accompanied by the paper it comes from. It is a 
study sandbox, not a production library.

The architecture follows the modern LLaMA-3 / Mistral stack rather than 
GPT-2 - meaning RoPE over learned positional embeddings, RMSNorm over 
LayerNorm, and SwiGLU over GELU FFN. These are the choices that matter 
in 2024-25 interviews and in production frontier models.

## Structure

01-tokenizers/     BPE (byte-level), WordPiece, Unigram LM - from scratch
02-attention/      Scaled dot-product, MHA, GQA - with and without RoPE  
03-positional/     Learned absolute, RoPE, ALiBi - side by side
04-normalization/  LayerNorm vs RMSNorm - derived and compared
05-ffn/            GELU FFN vs SwiGLU - with gating mechanism explained
06-training/       Backprop in numpy, AdamW, LR schedules, grad accumulation
07-peft/           LoRA and QLoRA from scratch, rank ablations
08-rlhf/           Reward model (Bradley-Terry), PPO with KL penalty

## How to use it

Each module is standalone - no dependencies between folders. Start with 
01-tokenizers and work forward. Every file has a __main__ block with a 
small test you can run to verify correctness.

## Companion repo

The clean, end-to-end trained model lives in nanogpt-modern - built by 
porting the best implementations from this repo into a fully runnable 
training pipeline.
