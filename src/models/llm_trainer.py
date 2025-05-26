"""LLM Trainer for instruction fine-tuning with QLoRA."""

import os
import logging
import torch
from typing import Dict, Any, Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

logger = logging.getLogger(__name__)

class LLMTrainer:
    """Trainer for instruction fine-tuning of LLMs with QLoRA.
    
    This class handles training, inference, and evaluation of LLMs
    for instruction-following tasks.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        quantization_config: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        training_args: Optional[Dict[str, Any]] = None
    ):
        """Initialize the LLMTrainer.
        
        Args:
            model_name: Name or path of the model to fine-tune
            output_dir: Directory to save outputs
            quantization_config: BitsAndBytes quantization config
            lora_config: LoRA configuration
            training_args: Training arguments
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.quantization_config = quantization_config or self._get_default_quantization_config()
        self.lora_config = lora_config or self._get_default_lora_config()
        self.training_args = training_args or {}
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _get_default_quantization_config(self) -> Dict[str, Any]:
        """Get default quantization configuration."""
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True
        }
    
    def _get_default_lora_config(self) -> Dict[str, Any]:
        """Get default LoRA configuration."""
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                             "gate_proj", "up_proj", "down_proj"]
        }
    
    def load_model_and_tokenizer(self, checkpoint_path: Optional[str] = None):
        """Load model and tokenizer.
        
        Args:
            checkpoint_path: Path to checkpoint for inference/evaluation
        """
        logger.info(f"Loading model: {checkpoint_path or self.model_name}")
        
        if checkpoint_path:
            # Load from checkpoint for inference
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            # Load base model for training
            bnb_config = BitsAndBytesConfig(**self.quantization_config)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="eager"
            )
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Add LoRA adapters
            peft_config = LoraConfig(**self.lora_config)
            self.model = get_peft_model(self.model, peft_config)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        max_seq_length: int = 2048,
        dataset_text_field: str = "text",
        packing: bool = True
    ):
        """Set up the SFTTrainer.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            max_seq_length: Maximum sequence length
            dataset_text_field: Field name containing the text
            packing: Whether to use packing for efficiency
        """
        # Set up training arguments
        default_training_args = {
            "output_dir": self.output_dir,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "optim": "paged_adamw_32bit",
            "learning_rate": 3e-5,
            "warmup_steps": 10,
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 100,
            "evaluation_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 100 if eval_dataset else None,
            "save_total_limit": 1,
            "load_best_model_at_end": True if eval_dataset else False,
            "report_to": "wandb" if os.getenv("WANDB_API_KEY") else "none",
            "group_by_length": True,
            "fp16": False,
            "bf16": False
        }
        
        # Merge with user-provided training arguments
        training_args_dict = {**default_training_args, **self.training_args}
        training_args = TrainingArguments(**training_args_dict)
        
        # Set up trainer
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=max_seq_length,
            dataset_text_field=dataset_text_field,
            packing=packing
        )
        
        logger.info("Trainer setup completed")
    
    def train(self):
        """Train the model."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer first.")
        
        logger.info("Starting training")
        train_result = self.trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return train_result
    
    def generate_predictions(
        self,
        test_dataset: Dataset,
        instruction_template: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.75,
        output_file: Optional[str] = None
    ) -> List[str]:
        """Generate predictions for test dataset.
        
        Args:
            test_dataset: Test dataset
            instruction_template: Template for formatting instructions
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            output_file: Path to save predictions
            
        Returns:
            List of generated predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer first.")
        
        logger.info("Generating predictions")
        self.model.eval()
        
        predictions = []
        references = []
        
        for i, sample in enumerate(test_dataset):
            # Format input with instruction template
            formatted_input = self._format_instruction(
                sample, instruction_template, include_output=False
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_response = self._extract_response(full_response, instruction_template)
            
            predictions.append(clean_response)
            if 'answer' in sample or 'abstractive_answer' in sample:
                references.append(sample.get('answer', sample.get('abstractive_answer', '')))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{len(test_dataset)} predictions")
        
        # Save predictions if output file specified
        if output_file:
            self._save_predictions(predictions, references, output_file)
        
        return predictions
    
    def _format_instruction(
        self,
        sample: Dict[str, Any],
        template: str,
        include_output: bool = True
    ) -> str:
        """Format sample into instruction template.
        
        Args:
            sample: Data sample
            template: Instruction template type
            include_output: Whether to include output in formatting
            
        Returns:
            Formatted instruction string
        """
        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', sample.get('abstractive_answer', ''))
        
        if template == "chatml":
            instruction = f"<|im_start|>system\nBạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam.<|im_end|>\n"
            instruction += f"<|im_start|>user\nDựa vào nội dung văn bản pháp luật sau:\n{context}\n"
            instruction += f"Bạn hãy đưa ra câu trả lời cho câu hỏi:\n{question}<|im_end|>\n"
            instruction += f"<|im_start|>assistant\n"
            if include_output:
                instruction += f"{answer}<|im_end|>"
                
        elif template == "alpaca":
            instruction = "Below is an instruction that describes a task, paired with an input that provides further context. "
            instruction += "Write a response that appropriately completes the request.\n\n"
            instruction += "### Instruction:\nDựa vào văn bản pháp luật được cung cấp, hãy trả lời câu hỏi sau.\n\n"
            instruction += f"### Input:\nContext: {context}\nQuestion: {question}\n\n"
            instruction += "### Response:\n"
            if include_output:
                instruction += answer
                
        elif template == "vicuna":
            instruction = f"USER: Dựa vào nội dung sau:\n{context}\n\nTrả lời câu hỏi: {question}\n"
            instruction += "ASSISTANT: "
            if include_output:
                instruction += answer
                
        else:
            # Custom template - user should provide format string
            instruction = template.format(
                context=context,
                question=question,
                answer=answer if include_output else ""
            )
        
        return instruction
    
    def _extract_response(self, full_response: str, template: str) -> str:
        """Extract clean response from generated text.
        
        Args:
            full_response: Full generated text including prompt
            template: Instruction template type
            
        Returns:
            Clean response text
        """
        if template == "chatml":
            # Extract text after last "assistant" marker
            if "<|im_start|>assistant" in full_response:
                response = full_response.split("<|im_start|>assistant")[-1]
                # Remove end token if present
                response = response.replace("<|im_end|>", "").strip()
            else:
                response = full_response
                
        elif template == "alpaca":
            # Extract text after "### Response:"
            if "### Response:" in full_response:
                response = full_response.split("### Response:")[-1].strip()
            else:
                response = full_response
                
        elif template == "vicuna":
            # Extract text after "ASSISTANT:"
            if "ASSISTANT:" in full_response:
                response = full_response.split("ASSISTANT:")[-1].strip()
            else:
                response = full_response
                
        else:
            # For custom templates, try to find common patterns
            # This is a simple heuristic and may need adjustment
            response = full_response.split("\n")[-1].strip()
        
        # Clean up any remaining special tokens
        special_tokens = ["<|im_start|>", "<|im_end|>", "<s>", "</s>", "[INST]", "[/INST]"]
        for token in special_tokens:
            response = response.replace(token, "")
        
        return response.strip()
    
    def _save_predictions(
        self,
        predictions: List[str],
        references: List[str],
        output_file: str
    ):
        """Save predictions to file.
        
        Args:
            predictions: List of predictions
            references: List of references
            output_file: Output file path
        """
        import pandas as pd
        
        df = pd.DataFrame({
            'predictions': predictions,
            'references': references
        })
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        elif output_file.endswith('.json'):
            df.to_json(output_file, orient='records', force_ascii=False, indent=2)
        else:
            # Default to CSV
            df.to_csv(f"{output_file}.csv", index=False, encoding='utf-8-sig')
        
        logger.info(f"Predictions saved to {output_file}")