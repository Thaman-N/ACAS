import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the Legal Contract Analysis System")
    parser.add_argument("--mode", choices=["agent", "finetune", "evaluate", "all"], required=True,
                        help="Which component to run")
    parser.add_argument("--model_path", default="./Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
                        help="Path to the model")
    args = parser.parse_args()

    if args.mode == "agent" or args.mode == "all":
        print("Starting multimodal agent server...")
        subprocess.Popen(["python", "multimodal_agent.py"])
        print("Agent server started at http://localhost:8000")
        print("Open multimodal-agent-frontend.html in your browser")

    if args.mode == "finetune" or args.mode == "all":
        print("Starting fine-tuning process...")
        subprocess.call([
            "python", "qlora-fine-tuning.py",
            "--base_model", args.model_path,
            "--dataset_path", "./processed_contracts.pkl",
            "--output_dir", "./legal-contract-model",
            "--cuad_path", "./CUAD_v1/CUAD_v1.json"
        ])

    if args.mode == "evaluate" or args.mode == "all":
        print("Running evaluation framework...")
        subprocess.call([
            "python", "evaluation-framework.py",
            "--model_path", args.model_path,
            "--evaluation_data", "./legal_evaluation_data.json",
            "--output_dir", "./evaluation_results",
            "--use_gpu"
        ])

if __name__ == "__main__":
    main()