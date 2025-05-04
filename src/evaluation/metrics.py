"""Metrics for the ViAG project."""

import re
import logging
import numpy as np
from typing import Dict, List, Any
import nltk
import spacy
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_metric
import evaluate

logger = logging.getLogger(__name__)

# Try to load Vietnamese Spacy model
try:
    nlp = spacy.load('vi_core_news_lg')
except Exception as e:
    logger.warning(f"Failed to load Spacy model: {e}. Using en_core_web_sm as fallback.")
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        logger.error("Failed to load any Spacy model.")
        nlp = None

# Try to load metrics
try:
    rouge = load_metric('rouge')
    meteor = evaluate.load('meteor')
    bertscore = evaluate.load("bertscore")
except Exception as e:
    logger.error(f"Failed to load metrics: {e}")
    rouge, meteor, bertscore = None, None, None

def format_qa(text: str) -> str:
    """Format QA text.
    
    Args:
        text (str): Text to format.
        
    Returns:
        str: Formatted text.
    """
    # Remove special tokens and non-alphanumeric characters
    text = re.sub(r'<unk>|<pad>|\</s>|<pad>\*|[^\w\s,.;:!?()\-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU score.
    
    Args:
        predictions (List[str]): List of predicted texts.
        references (List[str]): List of reference texts.
        
    Returns:
        Dict[str, float]: BLEU scores.
    """
    if nlp is None:
        logger.error("Spacy model not loaded. Cannot calculate BLEU score.")
        return {f"BLEU-{n}": 0.0 for n in range(1, 5)}
    
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    
    for pred, ref in zip(predictions, references):
        pred_tokens = [token.text for token in nlp(pred)]
        ref_tokens = [token.text for token in nlp(ref)]
        smoothing = SmoothingFunction().method1
        
        for n, weight in enumerate(weights, start=1):
            score = sentence_bleu([ref_tokens], pred_tokens, weights=weight, smoothing_function=smoothing)
            bleu_scores[n].append(score)
    
    return {f"BLEU-{n}": (sum(scores) / len(scores)) * 100 for n, scores in bleu_scores.items()}

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate metrics.
    
    Args:
        predictions (List[str]): List of predicted texts.
        references (List[str]): List of reference texts.
        
    Returns:
        Dict[str, float]: Metrics.
    """
    if rouge is None or meteor is None or bertscore is None:
        logger.error("Metrics not loaded. Cannot calculate metrics.")
        return {}
    
    results = {}
    
    # Calculate ROUGE scores
    try:
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        results.update({k: v.mid.fmeasure * 100 for k, v in rouge_scores.items()})
    except Exception as e:
        logger.error(f"Failed to calculate ROUGE scores: {e}")
    
    # Calculate METEOR score
    try:
        meteor_score = meteor.compute(predictions=predictions, references=references)['meteor'] * 100
        results['METEOR'] = meteor_score
    except Exception as e:
        logger.error(f"Failed to calculate METEOR score: {e}")
    
    # Calculate BERTScore
    try:
        bertscore_results = bertscore.compute(predictions=predictions, references=references, lang='vi')
        results['BERTScore-F1'] = np.mean(bertscore_results['f1']) * 100
    except Exception as e:
        logger.error(f"Failed to calculate BERTScore: {e}")
    
    return results

def compute_metrics_for_seq2seq(tokenizer):
    """Create a compute_metrics function for Seq2SeqTrainer.
    
    Args:
        tokenizer: The tokenizer used.
        
    Returns:
        function: A function to compute metrics.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode predictions and references
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Format text
        decoded_preds = [format_qa(p) for p in decoded_preds]
        decoded_labels = [format_qa(l) for l in decoded_labels]
        
        # Calculate BLEU scores
        bleu_scores = calculate_bleu(decoded_preds, decoded_labels)
        
        # Calculate other metrics
        other_metrics = calculate_metrics(decoded_preds, decoded_labels)
        
        # Combine metrics
        all_metrics = {**bleu_scores, **other_metrics}
        
        return all_metrics
    
    return compute_metrics