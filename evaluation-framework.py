import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datasets import Dataset
from typing import List, Dict, Any, Tuple, Optional, Union
import argparse
import time
import csv
from sentence_transformers import SentenceTransformer, util

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalContractEvaluator:
    """Evaluation framework for legal contract analysis models."""
    
    def __init__(
        self,
        model_path: str,
        reference_model_path: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        evaluation_data_path: Optional[str] = None,
        output_dir: str = "./evaluation_results",
        use_gpu: bool = True
    ):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the model to evaluate (fine-tuned model)
            reference_model_path: Path to reference model for comparison (optional)
            embedding_model_name: Name of the sentence embedding model to use
            evaluation_data_path: Path to evaluation dataset
            output_dir: Directory to save evaluation results
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path
        self.reference_model_path = reference_model_path
        self.embedding_model_name = embedding_model_name
        self.evaluation_data_path = evaluation_data_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model(s)
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=-1 if self.use_gpu else 0
            )
            logger.info(f"Main model loaded from {model_path}")
            
            if reference_model_path:
                self.reference_model = Llama(
                    model_path=reference_model_path,
                    n_ctx=4096,
                    n_gpu_layers=-1 if self.use_gpu else 0
                )
                logger.info(f"Reference model loaded from {reference_model_path}")
            else:
                self.reference_model = None
                logger.info("No reference model provided")
        except Exception as e:
            logger.error(f"Error loading model(s): {e}")
            raise
        
        # Load sentence embedding model for semantic similarity
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Embedding model loaded: {embedding_model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedding_model = None
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1  # BLEU smoothing
    
    def load_evaluation_data(self, data_path: Optional[str] = None) -> List[Dict]:
        """
        Load evaluation data
        
        Args:
            data_path: Path to evaluation data (JSON or CSV)
        
        Returns:
            List of evaluation examples
        """
        data_path = data_path or self.evaluation_data_path
        
        if not data_path:
            logger.warning("No evaluation data path provided")
            return []
        
        if not os.path.exists(data_path):
            logger.error(f"Evaluation data file not found: {data_path}")
            return []
        
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'examples' in data:
                    return data['examples']
                else:
                    logger.warning(f"Unexpected JSON format in {data_path}")
                    return []
                
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                
                # Convert DataFrame to list of dicts
                return df.to_dict('records')
                
            elif data_path.endswith('.pkl'):
                # For pickle files, assume it's a pandas DataFrame
                df = pd.read_pickle(data_path)
                
                # Process the DataFrame to create evaluation examples
                examples = []
                
                # Check if it's the processed contracts DataFrame
                if all(col in df.columns for col in ['contract_type', 'cleaned_text']):
                    # Sample contracts for evaluation
                    sample_size = min(50, len(df))
                    sampled_df = df.sample(sample_size, random_state=42)
                    
                    # Create different types of evaluation examples
                    for _, row in sampled_df.iterrows():
                        contract_text = row['cleaned_text'][:2000]  # Truncate for evaluation
                        contract_type = row['contract_type']
                        
                        # Contract type classification example
                        examples.append({
                            'task': 'classification',
                            'input': f"Identify the type of this legal contract:\n\n{contract_text}",
                            'reference': contract_type,
                            'metadata': {'contract_type': contract_type}
                        })
                        
                        # Contract analysis example
                        examples.append({
                            'task': 'analysis',
                            'input': f"Analyze the key terms and conditions in this contract:\n\n{contract_text}",
                            'reference': None,  # No reference for open-ended analysis
                            'metadata': {'contract_type': contract_type}
                        })
                        
                        # Entity extraction example
                        examples.append({
                            'task': 'extraction',
                            'input': f"Extract the parties involved in this contract:\n\n{contract_text}",
                            'reference': None,  # No reference for extraction
                            'metadata': {'contract_type': contract_type}
                        })
                    
                    logger.info(f"Created {len(examples)} evaluation examples from processed contracts")
                    return examples
                else:
                    logger.warning(f"Unrecognized DataFrame format in {data_path}")
                    return []
            else:
                logger.error(f"Unsupported file format: {data_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            return []
    
    def create_cuad_evaluation_data(self, cuad_path: str) -> List[Dict]:
        """
        Create evaluation data from CUAD dataset
        
        Args:
            cuad_path: Path to CUAD dataset JSON
        
        Returns:
            List of evaluation examples
        """
        if not os.path.exists(cuad_path):
            logger.error(f"CUAD file not found: {cuad_path}")
            return []
        
        try:
            with open(cuad_path, 'r') as f:
                cuad_data = json.load(f)
            
            examples = []
            
            # Process CUAD data
            for item in cuad_data.get('data', []):
                title = item.get('title', '')
                paragraphs = item.get('paragraphs', [])
                
                for para in paragraphs:
                    context = para.get('context', '')
                    qas = para.get('qas', [])
                    
                    for qa in qas:
                        question = qa.get('question', '')
                        answers = qa.get('answers', [])
                        
                        if answers and question:
                            # Get answer text
                            answer_text = answers[0].get('text', '')
                            
                            # Create QA example
                            examples.append({
                                'task': 'qa',
                                'input': f"Contract text:\n{context[:1500]}\n\nQuestion: {question}",
                                'reference': answer_text,
                                'metadata': {'document': title, 'question': question}
                            })
            
            logger.info(f"Created {len(examples)} evaluation examples from CUAD dataset")
            return examples
            
        except Exception as e:
            logger.error(f"Error processing CUAD data: {e}")
            return []
    
    def generate_completion(self, model, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a completion using the specified model
        
        Args:
            model: The model to use for generation
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text
        """
        try:
            response = model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            return ""
    
    def compute_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute evaluation metrics between prediction and reference
        
        Args:
            prediction: Model prediction
            reference: Reference answer
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Skip if reference is None (for open-ended tasks)
        if reference is None:
            return {'has_metrics': False}
        
        try:
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, prediction)
            metrics['rouge1_f'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2_f'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL_f'] = rouge_scores['rougeL'].fmeasure
            
            # BLEU score
            reference_tokens = reference.lower().split()
            prediction_tokens = prediction.lower().split()
            
            if reference_tokens:
                metrics['bleu'] = sentence_bleu([reference_tokens], prediction_tokens, 
                                               smoothing_function=self.smoothing)
            else:
                metrics['bleu'] = 0.0
            
            # Semantic similarity using sentence embeddings
            if self.embedding_model:
                reference_emb = self.embedding_model.encode(reference, convert_to_tensor=True)
                prediction_emb = self.embedding_model.encode(prediction, convert_to_tensor=True)
                
                similarity = util.pytorch_cos_sim(reference_emb, prediction_emb).item()
                metrics['semantic_similarity'] = float(similarity)
            
            metrics['has_metrics'] = True
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            metrics['error'] = str(e)
            metrics['has_metrics'] = False
        
        return metrics
    
    def evaluate_example(self, example: Dict) -> Dict:
        """
        Evaluate a single example
        
        Args:
            example: Evaluation example containing input and reference
        
        Returns:
            Dictionary with evaluation results
        """
        task = example.get('task', 'general')
        input_text = example.get('input', '')
        reference = example.get('reference', None)
        metadata = example.get('metadata', {})
        
        # Skip if no input
        if not input_text:
            return {
                'task': task,
                'input': input_text,
                'error': 'No input provided',
                'metadata': metadata
            }
        
        # Format prompt
        if task in ['qa', 'classification', 'extraction']:
            # For specific tasks, use the input directly
            prompt = input_text
        else:
            # For general analysis, add a system prompt
            prompt = f"You are a legal expert analyzing contracts. {input_text}"
        
        # Generate prediction with main model
        start_time = time.time()
        prediction = self.generate_completion(self.model, prompt)
        main_model_time = time.time() - start_time
        
        # Generate prediction with reference model if available
        reference_prediction = None
        reference_model_time = None
        
        if self.reference_model:
            start_time = time.time()
            reference_prediction = self.generate_completion(self.reference_model, prompt)
            reference_model_time = time.time() - start_time
        
        # Compute metrics if reference is available
        metrics = {}
        reference_metrics = {}
        
        if reference:
            metrics = self.compute_metrics(prediction, reference)
            
            if reference_prediction:
                reference_metrics = self.compute_metrics(reference_prediction, reference)
        
        # Return results
        result = {
            'task': task,
            'input': input_text,
            'reference': reference,
            'prediction': prediction,
            'metrics': metrics,
            'main_model_time': main_model_time,
            'metadata': metadata
        }
        
        if self.reference_model:
            result['reference_prediction'] = reference_prediction
            result['reference_metrics'] = reference_metrics
            result['reference_model_time'] = reference_model_time
        
        return result
    
    def evaluate(self, examples: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Evaluate the model on multiple examples
        
        Args:
            examples: List of evaluation examples
        
        Returns:
            List of evaluation results
        """
        # If no examples provided, load from data path
        if examples is None:
            examples = self.load_evaluation_data()
        
        if not examples:
            logger.warning("No evaluation examples available")
            return []
        
        logger.info(f"Starting evaluation on {len(examples)} examples")
        
        results = []
        for i, example in enumerate(tqdm(examples, desc="Evaluating")):
            try:
                result = self.evaluate_example(example)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating example {i}: {e}")
                results.append({
                    'task': example.get('task', 'unknown'),
                    'input': example.get('input', ''),
                    'error': str(e),
                    'metadata': example.get('metadata', {})
                })
        
        return results
    
    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze evaluation results
        
        Args:
            results: List of evaluation results
        
        Returns:
            Dictionary with analysis
        """
        if not results:
            return {'error': 'No results to analyze'}
        
        analysis = {
            'total_examples': len(results),
            'tasks': {},
            'metrics': {},
            'timings': {
                'main_model': [],
                'reference_model': []
            },
            'errors': []
        }
        
        # Count tasks
        task_counts = {}
        for result in results:
            task = result.get('task', 'unknown')
            task_counts[task] = task_counts.get(task, 0) + 1
        
        analysis['tasks'] = task_counts
        
        # Collect metrics
        metric_values = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'bleu': [],
            'semantic_similarity': []
        }
        
        reference_metric_values = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'bleu': [],
            'semantic_similarity': []
        }
        
        for result in results:
            # Skip results with errors
            if 'error' in result:
                analysis['errors'].append(result['error'])
                continue
            
            # Add timing information
            if 'main_model_time' in result:
                analysis['timings']['main_model'].append(result['main_model_time'])
            
            if 'reference_model_time' in result:
                analysis['timings']['reference_model'].append(result['reference_model_time'])
            
            # Skip if no metrics
            metrics = result.get('metrics', {})
            if not metrics.get('has_metrics', False):
                continue
            
            # Add metrics
            for metric_name in metric_values.keys():
                if metric_name in metrics:
                    metric_values[metric_name].append(metrics[metric_name])
            
            # Add reference model metrics if available
            reference_metrics = result.get('reference_metrics', {})
            if reference_metrics.get('has_metrics', False):
                for metric_name in reference_metric_values.keys():
                    if metric_name in reference_metrics:
                        reference_metric_values[metric_name].append(reference_metrics[metric_name])
        
        # Calculate average metrics
        for metric_name, values in metric_values.items():
            if values:
                analysis['metrics'][f'avg_{metric_name}'] = sum(values) / len(values)
                analysis['metrics'][f'min_{metric_name}'] = min(values)
                analysis['metrics'][f'max_{metric_name}'] = max(values)
        
        # Add reference model comparison if available
        if any(len(v) > 0 for v in reference_metric_values.values()):
            analysis['reference_comparison'] = {}
            
            for metric_name, values in reference_metric_values.items():
                if values:
                    avg_value = sum(values) / len(values)
                    analysis['reference_comparison'][f'avg_{metric_name}'] = avg_value
                    
                    # Calculate improvement over reference model
                    if f'avg_{metric_name}' in analysis['metrics']:
                        main_avg = analysis['metrics'][f'avg_{metric_name}']
                        diff = main_avg - avg_value
                        rel_diff = diff / avg_value if avg_value != 0 else 0
                        
                        analysis['reference_comparison'][f'diff_{metric_name}'] = diff
                        analysis['reference_comparison'][f'rel_diff_{metric_name}'] = rel_diff
        
        # Calculate average timings
        if analysis['timings']['main_model']:
            analysis['timings']['avg_main_model'] = sum(analysis['timings']['main_model']) / len(analysis['timings']['main_model'])
        
        if analysis['timings']['reference_model']:
            analysis['timings']['avg_reference_model'] = sum(analysis['timings']['reference_model']) / len(analysis['timings']['reference_model'])
        
        return analysis
    
    def generate_visualizations(self, results: List[Dict], analysis: Dict[str, Any]) -> None:
        """
        Generate visualizations from evaluation results
        
        Args:
            results: List of evaluation results
            analysis: Analysis dictionary
        """
        if not results or not analysis:
            logger.warning("No results or analysis to visualize")
            return
        
        # Create visualization directory
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Task distribution pie chart
        if analysis.get('tasks'):
            plt.figure(figsize=(10, 6))
            plt.pie(
                list(analysis['tasks'].values()), 
                labels=list(analysis['tasks'].keys()),
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title('Distribution of Evaluation Tasks')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "task_distribution.png"))
            plt.close()
        
        # 2. Metrics comparison bar chart
        if analysis.get('metrics') and analysis.get('reference_comparison'):
            plt.figure(figsize=(12, 8))
            
            metrics_to_plot = ['avg_rouge1_f', 'avg_rouge2_f', 'avg_rougeL_f', 'avg_bleu', 'avg_semantic_similarity']
            main_values = [analysis['metrics'].get(m, 0) for m in metrics_to_plot]
            ref_values = [analysis['reference_comparison'].get(m, 0) for m in metrics_to_plot]
            
            x = range(len(metrics_to_plot))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], main_values, width, label='Fine-tuned Model')
            plt.bar([i + width/2 for i in x], ref_values, width, label='Reference Model')
            
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.title('Metrics Comparison Between Models')
            plt.xticks(x, [m.replace('avg_', '') for m in metrics_to_plot])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "metrics_comparison.png"))
            plt.close()
        
        # 3. Response time comparison
        if analysis.get('timings') and 'avg_main_model' in analysis['timings'] and 'avg_reference_model' in analysis['timings']:
            plt.figure(figsize=(8, 6))
            
            models = ['Fine-tuned Model', 'Reference Model']
            times = [analysis['timings']['avg_main_model'], analysis['timings']['avg_reference_model']]
            
            plt.bar(models, times)
            plt.xlabel('Model')
            plt.ylabel('Average Response Time (seconds)')
            plt.title('Response Time Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "response_time.png"))
            plt.close()
        
        # 4. Semantic similarity histogram
        semantic_sim_values = []
        for result in results:
            metrics = result.get('metrics', {})
            if 'semantic_similarity' in metrics:
                semantic_sim_values.append(metrics['semantic_similarity'])
        
        if semantic_sim_values:
            plt.figure(figsize=(10, 6))
            plt.hist(semantic_sim_values, bins=20, alpha=0.7)
            plt.xlabel('Semantic Similarity Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Semantic Similarity Scores')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "semantic_similarity_distribution.png"))
            plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def run_evaluation_pipeline(self, examples: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline
        
        Args:
            examples: List of evaluation examples (optional)
        
        Returns:
            Dictionary with evaluation summary
        """
        # Load or use provided examples
        if examples is None:
            examples = self.load_evaluation_data()
        
        if not examples:
            return {'error': 'No evaluation examples available'}
        
        # Run evaluation
        results = self.evaluate(examples)
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Generate visualizations
        self.generate_visualizations(results, analysis)
        
        # Save results
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        analysis_path = os.path.join(self.output_dir, "evaluation_analysis.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        logger.info(f"Evaluation analysis saved to {analysis_path}")
        
        return {
            'num_examples': len(examples),
            'num_results': len(results),
            'analysis': analysis,
            'results_path': results_path,
            'analysis_path': analysis_path
        }

def create_evaluation_data(cuad_path: str, output_path: str):
    """
    Create evaluation dataset from CUAD dataset
    
    Args:
        cuad_path: Path to CUAD dataset
        output_path: Path to save evaluation dataset
    """
    try:
        # Check CUAD path
        if not os.path.exists(cuad_path):
            logger.error(f"CUAD dataset not found at {cuad_path}")
            return
        
        # Load CUAD data
        with open(cuad_path, 'r') as f:
            cuad_data = json.load(f)
        
        # Create evaluation examples
        examples = []
        
        # Process CUAD data
        for item in cuad_data.get('data', []):
            title = item.get('title', '')
            paragraphs = item.get('paragraphs', [])
            
            for para in paragraphs:
                context = para.get('context', '')
                qas = para.get('qas', [])
                
                # Skip if no context
                if not context:
                    continue
                
                # Create classification example
                examples.append({
                    'task': 'classification',
                    'input': f"Identify the type of this legal contract:\n\n{context[:2000]}",
                    'reference': 'Legal Contract',  # Generic label since we don't have fine-grained types
                    'metadata': {'document': title}
                })
                
                # Create extraction examples
                examples.append({
                    'task': 'extraction',
                    'input': f"Extract the parties involved in this contract:\n\n{context[:2000]}",
                    'reference': None,  # No reference for extraction
                    'metadata': {'document': title}
                })
                
                # Create QA examples from CUAD annotations
                for qa in qas:
                    question = qa.get('question', '')
                    answers = qa.get('answers', [])
                    
                    if answers and question:
                        # Get answer text
                        answer_text = answers[0].get('text', '')
                        
                        # Create QA example
                        examples.append({
                            'task': 'qa',
                            'input': f"Contract text:\n{context[:1500]}\n\nQuestion: {question}",
                            'reference': answer_text,
                            'metadata': {'document': title, 'question': question}
                        })
        
        # Add special legal reasoning examples
        legal_reasoning_examples = [
            {
                'task': 'reasoning',
                'input': "Analyze the following force majeure clause and explain its implications:\n\n\"Neither party shall be liable for any failure or delay in performance of its obligations under this Agreement to the extent such failure or delay is caused by a force majeure event, including but not limited to acts of God, natural disasters, war, civil unrest, labor disputes, or government actions, that is beyond the reasonable control of such party.\"",
                'reference': "This force majeure clause excuses parties from performance when prevented by extraordinary events outside their control. Key points: it covers specific events, requires events to be beyond reasonable control, only excuses performance to the extent prevented, and implicitly requires resumption once the event ends.",
                'metadata': {'type': 'force_majeure'}
            },
            {
                'task': 'reasoning',
                'input': "Identify potential issues with this non-compete clause:\n\n\"Employee agrees not to engage in any business activities that compete with the Company anywhere in the world for a period of 5 years after termination of employment.\"",
                'reference': "This non-compete clause has several enforceability issues: unreasonable geographic scope (worldwide), excessive duration (5 years), vague prohibited activities, and lacks consideration. Most jurisdictions require non-competes to be reasonable in scope, duration, and geography.",
                'metadata': {'type': 'non_compete'}
            }
        ]
        
        # Add legal reasoning examples
        examples.extend(legal_reasoning_examples)
        
        # Save evaluation data
        with open(output_path, 'w') as f:
            json.dump({'examples': examples}, f, indent=2)
        
        logger.info(f"Created {len(examples)} evaluation examples")
        logger.info(f"Evaluation dataset saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating evaluation data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a legal contract analysis model")
    
    # Main arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model to evaluate")
    parser.add_argument("--reference_model", type=str, default=None,
                       help="Path to reference model for comparison")
    parser.add_argument("--evaluation_data", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--use_gpu", action="store_true", default=True,
                       help="Use GPU for inference")
    
    # Dataset creation mode
    parser.add_argument("--create_dataset", action="store_true",
                       help="Create evaluation dataset instead of running evaluation")
    parser.add_argument("--cuad_path", type=str, default=None,
                       help="Path to CUAD dataset for creating evaluation data")
    parser.add_argument("--dataset_output", type=str, default="./evaluation_dataset.json",
                       help="Output path for created evaluation dataset")
    
    args = parser.parse_args()
    
    # If in dataset creation mode
    if args.create_dataset:
        if not args.cuad_path:
            logger.error("CUAD path must be provided with --cuad_path when using --create_dataset")
            return
        
        create_evaluation_data(args.cuad_path, args.dataset_output)
        return
    
    # Otherwise, run evaluation
    evaluator = LegalContractEvaluator(
        model_path=args.model_path,
        reference_model_path=args.reference_model,
        evaluation_data_path=args.evaluation_data,
        output_dir=args.output_dir,
        use_gpu=args.use_gpu
    )
    
    # Run evaluation pipeline
    result = evaluator.run_evaluation_pipeline()
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Number of examples evaluated: {result.get('num_examples', 0)}")
    print(f"Results saved to: {result.get('results_path', 'unknown')}")
    
    # Print metrics summary if available
    analysis = result.get('analysis', {})
    if 'metrics' in analysis:
        print("\nMetrics:")
        for metric_name, value in analysis['metrics'].items():
            if metric_name.startswith('avg_'):
                print(f"  {metric_name.replace('avg_', '')}: {value:.4f}")
    
    # Print comparison with reference model if available
    if 'reference_comparison' in analysis:
        print("\nComparison with Reference Model:")
        for metric_name, value in analysis['reference_comparison'].items():
            if metric_name.startswith('rel_diff_'):
                metric = metric_name.replace('rel_diff_', '')
                percentage = value * 100
                direction = "improvement" if percentage > 0 else "decrease"
                print(f"  {metric}: {abs(percentage):.2f}% {direction}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
