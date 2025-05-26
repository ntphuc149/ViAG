"""Instruction preprocessor for LLM fine-tuning."""

import logging
from typing import Dict, List, Optional, Union
from datasets import Dataset
import pandas as pd

logger = logging.getLogger(__name__)

class InstructionPreprocessor:
    """Preprocessor for instruction-based fine-tuning of LLMs.
    
    This class handles formatting data into instruction templates
    for training language models.
    """
    
    def __init__(
        self,
        instruction_template: str = "chatml",
        system_prompt: Optional[str] = None,
        max_length: Optional[int] = None
    ):
        """Initialize the InstructionPreprocessor.
        
        Args:
            instruction_template: Template format (chatml, alpaca, vicuna, or custom)
            system_prompt: System prompt to use (if applicable)
            max_length: Maximum length for truncation
        """
        self.instruction_template = instruction_template
        self.system_prompt = system_prompt
        self.max_length = max_length
        
        # Default system prompts for different domains
        self.default_system_prompts = {
            "legal": "Bạn là chuyên gia về lĩnh vực pháp luật tại Việt Nam.",
            "medical": "Bạn là chuyên gia y tế có kiến thức chuyên sâu.",
            "general": "Bạn là trợ lý AI hữu ích, trung thực và không gây hại.",
            "educational": "Bạn là giáo viên nhiệt tình và kiên nhẫn."
        }
    
    def prepare_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        text_column: Optional[str] = None,
        context_column: str = "context",
        question_column: str = "question",
        answer_column: str = "answer",
        include_output: bool = True
    ) -> Dataset:
        """Prepare dataset for instruction fine-tuning.
        
        Args:
            dataset: Input dataset
            text_column: Column containing pre-formatted text (if available)
            context_column: Column containing context
            question_column: Column containing questions
            answer_column: Column containing answers
            include_output: Whether to include output in formatting (for training)
            
        Returns:
            Processed dataset with instruction formatting
        """
        # Convert to Dataset if needed
        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset, preserve_index=False)
        
        # Check for abstractive_answer column (compatibility with your data)
        if answer_column not in dataset.column_names and 'abstractive_answer' in dataset.column_names:
            answer_column = 'abstractive_answer'
        
        def format_example(example):
            """Format a single example."""
            if text_column and text_column in example:
                # Use pre-formatted text if available
                return {"text": example[text_column]}
            
            # Format using template
            formatted_text = self._format_instruction(
                context=example.get(context_column, ""),
                question=example.get(question_column, ""),
                answer=example.get(answer_column, "") if include_output else "",
                include_output=include_output
            )
            
            return {"text": formatted_text}
        
        # Apply formatting
        formatted_dataset = dataset.map(
            format_example,
            remove_columns=[col for col in dataset.column_names if col not in ["text"]],
            desc="Formatting instructions"
        )
        
        logger.info(f"Prepared {len(formatted_dataset)} examples with {self.instruction_template} template")
        
        return formatted_dataset
    
    def _format_instruction(
        self,
        context: str,
        question: str,
        answer: str = "",
        include_output: bool = True
    ) -> str:
        """Format instruction based on template.
        
        Args:
            context: Context text
            question: Question text
            answer: Answer text
            include_output: Whether to include answer
            
        Returns:
            Formatted instruction string
        """
        # Get system prompt
        system_prompt = self.system_prompt or self.default_system_prompts.get("general", "")
        
        if self.instruction_template == "chatml":
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            text += f"<|im_start|>user\nDựa vào nội dung sau:\n{context}\n"
            text += f"Hãy trả lời câu hỏi: {question}<|im_end|>\n"
            text += f"<|im_start|>assistant\n"
            if include_output and answer:
                text += f"{answer}<|im_end|>"
                
        elif self.instruction_template == "alpaca":
            text = "Below is an instruction that describes a task, paired with an input that provides further context. "
            text += "Write a response that appropriately completes the request.\n\n"
            text += f"### Instruction:\n{question}\n\n"
            if context:
                text += f"### Input:\n{context}\n\n"
            text += "### Response:\n"
            if include_output and answer:
                text += answer
                
        elif self.instruction_template == "vicuna":
            text = f"USER: {system_prompt}\n" if system_prompt else ""
            text += f"Context: {context}\n" if context else ""
            text += f"Question: {question}\n"
            text += "ASSISTANT: "
            if include_output and answer:
                text += answer
                
        elif self.instruction_template == "llama":
            text = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n" if system_prompt else "[INST] "
            if context:
                text += f"Context: {context}\n"
            text += f"{question} [/INST] "
            if include_output and answer:
                text += answer
                
        else:
            # Custom template - should contain {context}, {question}, {answer} placeholders
            if isinstance(self.instruction_template, str) and "{" in self.instruction_template:
                text = self.instruction_template.format(
                    system=system_prompt,
                    context=context,
                    question=question,
                    answer=answer if include_output else ""
                )
            else:
                # Fallback to simple format
                text = f"Context: {context}\nQuestion: {question}\nAnswer: "
                if include_output and answer:
                    text += answer
        
        # Truncate if max_length is set
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def split_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, Dataset]:
        """Split dataset into train/val/test sets.
        
        Args:
            dataset: Input dataset
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            seed: Random seed
            
        Returns:
            Dictionary with train, val, and test datasets
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Convert to Dataset if needed
        if isinstance(dataset, pd.DataFrame):
            dataset = Dataset.from_pandas(dataset, preserve_index=False)
        
        # Calculate sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # Shuffle and split
        dataset = dataset.shuffle(seed=seed)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        logger.info(f"Split dataset: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        return {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
    
    def create_chat_format(self, tokenizer):
        """Apply chat format to tokenizer if needed.
        
        Args:
            tokenizer: Tokenizer to modify
            
        Returns:
            Modified tokenizer
        """
        # Check if tokenizer already has chat template
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            return tokenizer
        
        # Add special tokens for ChatML format if needed
        if self.instruction_template == "chatml":
            special_tokens = ["<|im_start|>", "<|im_end|>"]
            existing_tokens = tokenizer.get_vocab()
            tokens_to_add = [token for token in special_tokens if token not in existing_tokens]
            
            if tokens_to_add:
                tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
                logger.info(f"Added special tokens: {tokens_to_add}")
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer