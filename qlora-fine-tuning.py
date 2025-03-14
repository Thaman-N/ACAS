import os
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from tqdm import tqdm
import bitsandbytes as bnb
import torch.nn as nn
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LegalContractFineTuner:
    """Fine-tune a language model on legal contract data using QLoRA."""
    
    def __init__(
        self,
        base_model_name="meta-llama/Llama-3-8B",
        dataset_path="./processed_contracts.pkl",
        output_dir="./fine_tuned_model",
        cuad_json_path="./CUAD_v1/CUAD_v1.json"
    ):
        """
        Initialize the fine-tuner
        
        Args:
            base_model_name: Name or path of the base model
            dataset_path: Path to processed contracts
            output_dir: Directory to save fine-tuned model
            cuad_json_path: Path to CUAD JSON for fine-tuning examples
        """
        self.base_model_name = base_model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.cuad_json_path = cuad_json_path
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cpu":
            logger.warning("CUDA not available. Fine-tuning on CPU will be extremely slow.")
    
    def load_and_prepare_data(self):
        """Load and prepare data for fine-tuning."""
        # Load processed contracts
        logger.info(f"Loading processed contracts from {self.dataset_path}")
        try:
            contracts_df = pd.read_pickle(self.dataset_path)
            logger.info(f"Loaded {len(contracts_df)} contracts")
        except Exception as e:
            logger.error(f"Error loading contracts: {e}")
            return None
        
        # Load CUAD dataset for QA examples
        logger.info(f"Loading CUAD dataset from {self.cuad_json_path}")
        cuad_qa_examples = []
        
        try:
            with open(self.cuad_json_path, 'r') as f:
                cuad_data = json.load(f)
            
            # Process CUAD data to extract QA pairs
            for item in cuad_data.get('data', []):
                title = item.get('title', '')
                paragraphs = item.get('paragraphs', [])
                
                for para in paragraphs:
                    context = para.get('context', '')
                    qas = para.get('qas', [])
                    
                    for qa in qas:
                        question = qa.get('question', '')
                        answers = qa.get('answers', [])
                        
                        if answers:
                            # Use the first answer
                            answer_text = answers[0].get('text', '')
                            
                            if answer_text and question:
                                cuad_qa_examples.append({
                                    'contract': context,
                                    'question': question,
                                    'answer': answer_text
                                })
            
            logger.info(f"Extracted {len(cuad_qa_examples)} QA examples from CUAD")
        except Exception as e:
            logger.error(f"Error loading CUAD data: {e}")
            logger.info("Will proceed without QA examples")
            cuad_qa_examples = []
        
        # Create instruction dataset for fine-tuning
        instructions = []
        
        # 1. Contract Classification Instructions
        for i, row in contracts_df.sample(min(100, len(contracts_df))).iterrows():
            contract_text = row['cleaned_text'][:1000]  # Truncate for instruction examples
            contract_type = row['contract_type']
            
            instructions.append({
                'input': f"Identify the type of this legal contract:\n\n{contract_text}",
                'output': f"This is a {contract_type} contract."
            })
        
        # 2. Contract Analysis Instructions
        for i, row in contracts_df.sample(min(100, len(contracts_df))).iterrows():
            contract_text = row['cleaned_text'][:1500]  # Truncate for instruction examples
            
            instructions.append({
                'input': f"Analyze the key terms and conditions in this contract:\n\n{contract_text}",
                'output': "Based on my analysis of this contract, I've identified the following key elements:\n\n1. Parties involved: [would be extracted from the actual contract]\n2. Contract duration: [would be extracted from the actual contract]\n3. Key obligations: [would be extracted from the actual contract]\n4. Termination conditions: [would be extracted from the actual contract]\n5. Governing law: [would be extracted from the actual contract]"
            })
        
        # 3. Contract QA Instructions from CUAD
        for example in cuad_qa_examples[:200]:  # Limit to 200 examples
            instructions.append({
                'input': f"Contract text:\n{example['contract'][:1500]}\n\nQuestion: {example['question']}",
                'output': example['answer']
            })
        
        # 4. Legal Reasoning Instructions
        legal_reasoning_examples = [
            {
                'input': "Analyze the following force majeure clause and explain its implications:\n\n\"Neither party shall be liable for any failure or delay in performance of its obligations under this Agreement to the extent such failure or delay is caused by a force majeure event, including but not limited to acts of God, natural disasters, war, civil unrest, labor disputes, or government actions, that is beyond the reasonable control of such party.\"",
                'output': "This force majeure clause excuses parties from liability when performance is prevented by extraordinary events outside their control. Key implications:\n\n1. It covers specific events (acts of God, natural disasters, war, civil unrest, labor disputes, government actions)\n2. The event must be beyond 'reasonable control' of the party\n3. It only excuses performance 'to the extent' prevented by the force majeure event\n4. It suspends rather than terminates obligations\n5. It does not explicitly require notice to the other party\n6. It doesn't specify when performance must resume after the event ends\n\nThis clause protects parties from unexpected circumstances but leaves some ambiguity around notice requirements and resumption of obligations."
            },
            {
                'input': "Identify potential issues with this non-compete clause:\n\n\"Employee agrees not to engage in any business activities that compete with the Company anywhere in the world for a period of 5 years after termination of employment.\"",
                'output': "This non-compete clause has several potential issues that may affect its enforceability:\n\n1. Unreasonable geographic scope: 'anywhere in the world' is overly broad and likely unenforceable in many jurisdictions\n2. Excessive duration: 5 years is longer than what many courts consider reasonable\n3. Vague prohibited activities: 'any business activities that compete' is ambiguous and overly broad\n4. No consideration: the clause doesn't mention what the employee receives in exchange\n5. No legitimate business interest: fails to articulate what specific interest is being protected\n6. Potential violation of local laws: many jurisdictions limit or prohibit non-compete agreements\n\nCourts typically require non-compete clauses to be reasonable in scope, geography, and duration while protecting legitimate business interests. This clause likely fails those tests in most jurisdictions."
            }
        ]
        
        instructions.extend(legal_reasoning_examples)
        
        # Convert to HF dataset
        logger.info(f"Created {len(instructions)} instruction examples")
        dataset = Dataset.from_pandas(pd.DataFrame(instructions))
        
        return dataset
    
    def format_instruction(self, example):
        """Format a single instruction example as a prompt."""
        return f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        formatted_examples = [self.format_instruction({"input": inp, "output": out}) 
                             for inp, out in zip(examples["input"], examples["output"])]
        return self.tokenizer(
            formatted_examples,
            truncation=True,
            max_length=2048,
            padding="max_length"
        )
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with QLoRA configurations."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # BitsAndBytes configuration for quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with quantization config
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            return model, None
        
        return model, tokenizer
    
    def prepare_for_qlora(self, model):
        """Prepare the model for QLoRA fine-tuning."""
        logger.info("Preparing model for QLoRA fine-tuning")
        
        # Prepare the model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,                   # Rank dimension
            lora_alpha=32,          # Alpha parameter for LoRA scaling
            target_modules=[        # Target modules to apply LoRA
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,      # Dropout probability for LoRA layers
            bias="none",            # Don't train bias parameters
            task_type=TaskType.CAUSAL_LM  # Task type
        )
        
        # Get the PEFT model
        model = get_peft_model(model, lora_config)
        logger.info(f"Trainable parameters: {self.get_trainable_params(model)}")
        
        return model
    
    def get_trainable_params(self, model):
        """Get the number of trainable parameters in the model."""
        trainable_params = 0
        all_param = 0
        
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    
    def train(self):
        """Train the model using QLoRA."""
        # Load and prepare data
        dataset = self.load_and_prepare_data()
        if dataset is None:
            logger.error("Failed to load dataset. Aborting.")
            return
        
        # Split the dataset
        dataset_split = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        if model is None or tokenizer is None:
            logger.error("Failed to load model or tokenizer. Aborting.")
            return
        
        self.tokenizer = tokenizer
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(self.tokenize_function, batched=True)
        
        # Prepare model for QLoRA
        model = self.prepare_for_qlora(model)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",
            push_to_hub=False
        )
        
        # Create Trainer
        from transformers import Trainer
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        # Train model
        logger.info("Starting QLoRA fine-tuning")
        trainer.train()
        
        # Save the fine-tuned model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        trainer.save_model(f"{self.output_dir}/final")
        
        return model
    
    def test_model(self, test_inputs):
        """Test the fine-tuned model."""
        logger.info("Testing fine-tuned model")
        
        # Load the fine-tuned model
        adapter_path = f"{self.output_dir}/final"
        
        if not os.path.exists(adapter_path):
            logger.error(f"Fine-tuned model not found at {adapter_path}")
            return
        
        # BitsAndBytes configuration for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load the base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load the adapter weights (LoRA)
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        # Set up generation pipeline
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Test inputs
        results = []
        for i, test_input in enumerate(test_inputs):
            logger.info(f"Testing input {i+1}/{len(test_inputs)}")
            prompt = f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n"
            
            result = generation_pipeline(prompt)[0]['generated_text']
            
            # Extract just the assistant's response
            assistant_response = result.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
            
            results.append({
                "input": test_input,
                "output": assistant_response
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model on legal contract data using QLoRA")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3-8B", 
                        help="Base model to fine-tune")
    parser.add_argument("--dataset_path", type=str, default="./processed_contracts.pkl",
                       help="Path to processed contracts dataset")
    parser.add_argument("--output_dir", type=str, default="./legal_contract_model",
                       help="Directory to save fine-tuned model")
    parser.add_argument("--cuad_path", type=str, default="./CUAD_v1/CUAD_v1.json",
                       help="Path to CUAD dataset JSON")
    parser.add_argument("--test", action="store_true",
                       help="Test the model after fine-tuning")
    
    args = parser.parse_args()
    
    # Create fine-tuner
    fine_tuner = LegalContractFineTuner(
        base_model_name=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        cuad_json_path=args.cuad_path
    )
    
    # Train model
    model = fine_tuner.train()
    
    # Test if requested
    if args.test:
        test_inputs = [
            "What are the key elements of a non-compete clause?",
            "Identify the type of this contract: This License Agreement is made and entered into as of the date of last signature below (the \"Effective Date\") by and between Acme Corp, a Delaware corporation, with offices at 123 Main St, Dover, DE 19901 (\"Licensor\") and XYZ Inc., a California corporation, with offices at 456 Oak Ave, San Francisco, CA 94102 (\"Licensee\").",
            "Analyze this force majeure clause: In the event either party is unable to perform its obligations under the terms of this Agreement because of acts of God, strikes, equipment or transmission failure or damage reasonably beyond its control, or other causes reasonably beyond its control, such party shall not be liable for damages to the other for any damages resulting from such failure to perform or otherwise from such causes."
        ]
        
        results = fine_tuner.test_model(test_inputs)
        
        # Print results
        for i, result in enumerate(results):
            print(f"\nTest {i+1}:")
            print(f"Input: {result['input'][:100]}...")
            print(f"Output: {result['output']}")

if __name__ == "__main__":
    main()
