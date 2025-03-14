import json
import pandas as pd
from llama_cpp import Llama
import os
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

class LegalContractAnalyzer:
    def __init__(self, model_path):
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window
                n_gpu_layers=-1
            )
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Initializing with limited functionality (no LLM).")
            self.model = None
    
    def chain_of_thought_analysis(self, contract_text, max_length=1024):
        if self.model is None:
            return {"error": "Model not loaded", "contract_excerpt": contract_text[:100] + "..."}
            
        if len(contract_text) > max_length:
            contract_text = contract_text[:max_length]
        
        cot_prompt = f"""
        You are a highly skilled legal analyst. Analyze the following contract excerpt step by step.
        
        CONTRACT TEXT:
        {contract_text}
        
        Please follow these steps:
        1) First, identify what type of contract this is (e.g., employment, sales, lease).
        2) Next, identify the main parties involved in this contract.
        3) Then, identify key dates mentioned (execution date, effective date, termination date).
        4) Next, extract main obligations for each party.
        5) Identify any conditions or contingencies.
        6) Look for liability clauses and limitations.
        7) Finally, summarize the main purpose and scope of this contract.
        
        For each step, show your reasoning before giving your conclusion.
        """
        
        try:
            response = self.model.create_completion(
                prompt=cot_prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            analysis_text = response['choices'][0]['text']
            
            return {
                "cot_analysis": analysis_text,
                "contract_excerpt": contract_text[:100] + "..." # Just for reference
            }
        except Exception as e:
            print(f"Error during Chain of Thought analysis: {e}")
            return {
                "error": str(e),
                "contract_excerpt": contract_text[:100] + "..."
            }
    
    def tree_of_thought_analysis(self, contract_text, clause_type, max_length=1024):
        if self.model is None:
            return {"error": "Model not loaded", "clause_type": clause_type, "contract_excerpt": contract_text[:100] + "..."}
            
        if len(contract_text) > max_length:
            contract_text = contract_text[:max_length]
        
        tot_prompt = f"""
        You are a legal expert analyzing a contract. For the following contract excerpt, 
        focus on {clause_type.upper()} clauses and explore different interpretations.
        
        CONTRACT TEXT:
        {contract_text}
        
        Please analyze by exploring multiple possible interpretations:
        
        BRANCH 1: Standard interpretation
        - What is the standard legal interpretation of the {clause_type} clause?
        - How would this typically be enforced?
        
        BRANCH 2: Pro-first party interpretation
        - How might the first party interpret this clause to their advantage?
        - What legal arguments support this view?
        
        BRANCH 3: Pro-second party interpretation
        - How might the second party interpret this clause differently?
        - What legal arguments support their perspective?
        
        BRANCH 4: Potential ambiguities
        - Are there any ambiguities in the language?
        - How might these be resolved in a dispute?
        
        For each branch, give your analysis and reasoning step by step.
        """
        
        try:
            response = self.model.create_completion(
                prompt=tot_prompt,
                max_tokens=2048,
                temperature=0.2,
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            analysis_text = response['choices'][0]['text']
            
            return {
                "tot_analysis": analysis_text,
                "clause_type": clause_type,
                "contract_excerpt": contract_text[:100] + "..."
            }
        except Exception as e:
            print(f"Error during Tree of Thought analysis: {e}")
            return {
                "error": str(e),
                "clause_type": clause_type,
                "contract_excerpt": contract_text[:100] + "..."
            }
    
    def graph_of_thought_analysis(self, contract_text, max_length=1024):
        if self.model is None:
            return {"error": "Model not loaded", "contract_excerpt": contract_text[:100] + "..."}
            
        if len(contract_text) > max_length:
            contract_text = contract_text[:max_length]
        
        got_prompt = f"""
        You are a legal expert creating a conceptual graph of a contract.
        
        CONTRACT TEXT:
        {contract_text}
        
        Please identify the key elements of this contract and their relationships.
        Format your response as a JSON object with the following structure:
        
        {{
            "nodes": [
                {{"id": "1", "label": "Party A", "type": "party"}},
                {{"id": "2", "label": "Party B", "type": "party"}},
                {{"id": "3", "label": "Payment Obligation", "type": "obligation"}},
                ...
            ],
            "edges": [
                {{"source": "1", "target": "3", "label": "is obligated to"}},
                {{"source": "3", "target": "2", "label": "benefits"}},
                ...
            ]
        }}
        
        Node types can include: party, obligation, right, condition, date, asset, liability, clause.
        Edge labels should describe the relationship between nodes.
        
        Focus on creating a comprehensive but clear graph that shows how different elements of the contract relate to each other.
        """
        
        try:
            response = self.model.create_completion(
                prompt=got_prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["</s>"],
                echo=False
            )
            
            analysis_text = response['choices'][0]['text']
            
            try:
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = analysis_text[start_idx:end_idx]
                    graph_data = json.loads(json_str)
                else:
                    graph_data = {"error": "Could not extract JSON", "raw": analysis_text}
            
            except json.JSONDecodeError:
                graph_data = {"error": "JSON parsing error", "raw": analysis_text}
            
            return {
                "graph_data": graph_data,
                "contract_excerpt": contract_text[:100] + "..."
            }
        except Exception as e:
            print(f"Error during Graph of Thought analysis: {e}")
            return {
                "error": str(e),
                "contract_excerpt": contract_text[:100] + "..."
            }
    
    def visualize_contract_graph(self, graph_data, output_file="contract_graph.png"):
        if "error" in graph_data:
            print(f"Error: {graph_data['error']}")
            return
        
        try:
            G = nx.DiGraph()
            for node in graph_data.get("nodes", []):
                G.add_node(node["id"], label=node["label"], node_type=node.get("type", "unknown"))
            
            for edge in graph_data.get("edges", []):
                G.add_edge(edge["source"], edge["target"], label=edge.get("label", ""))
            
            pos = nx.spring_layout(G, seed=42)
            
            plt.figure(figsize=(14, 10))
            
            node_types = nx.get_node_attributes(G, 'node_type')
            color_map = {
                "party": "lightblue",
                "obligation": "lightgreen",
                "right": "lightyellow",
                "condition": "lightcoral",
                "date": "lightgrey",
                "asset": "lightpink",
                "liability": "orange",
                "clause": "purple"
            }
            
            default_color = "white"
            
            for node_type, color in color_map.items():
                nodes = [n for n, t in node_types.items() if t == node_type]
                if nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=700)
            
            unknown_nodes = [n for n in G.nodes() if node_types.get(n, "unknown") not in color_map]
            if unknown_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=unknown_nodes, node_color=default_color, node_size=700)
            
            nx.draw_networkx_edges(G, pos, arrows=True)
            
            node_labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
            
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                         markersize=10, label=node_type)
                              for node_type, color in color_map.items()]
            
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.axis('off')
            
            plt.title("Contract Relationship Graph", fontsize=16)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Graph visualization saved to {output_file}")
        except Exception as e:
            print(f"Error visualizing graph: {e}")

def analyze_specific_clause_type(analyzer, contracts_df, clause_type):
    matching_contracts = []
    
    keywords = {
        "termination": ["terminat", "cancel", "end the agreement"],
        "confidentiality": ["confidential", "non-disclosure", "secret"],
        "indemnification": ["indemnif", "hold harmless", "defend"],
        "liability": ["limit liability", "no liability", "not be liable"],
        "intellectual_property": ["intellectual property", "copyright", "patent"]
    }
    
    search_terms = keywords.get(clause_type.lower(), [clause_type])
    
    for idx, row in contracts_df.iterrows():
        text = row['cleaned_text'].lower()
        if any(term.lower() in text for term in search_terms):
            matching_contracts.append((idx, row))
    
    if not matching_contracts:
        print(f"No contracts found containing {clause_type} clauses")
        return None
    
    print(f"Found {len(matching_contracts)} contracts containing {clause_type} clauses")
    
    contract_idx, contract = matching_contracts[0]
    contract_text = contract['cleaned_text']
    
    print(f"Analyzing {clause_type} clause in contract: {contract['file_path']}")
    
    tot_result = analyzer.tree_of_thought_analysis(contract_text, clause_type)
    
    return tot_result

def main():
    model_path = "./Mistral-7B-Instruct-v0.3.Q5_K_M.gguf"
    
    try:
        analyzer = LegalContractAnalyzer(model_path)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        print("Continuing with limited functionality...")
        analyzer = None
    
    try:
        contracts_df = pd.read_pickle("processed_contracts.pkl")
        print(f"Loaded {len(contracts_df)} processed contracts")
    except FileNotFoundError:
        print("Error: Run Phase 1 script first to generate processed_contracts.pkl")
        return
    except Exception as e:
        print(f"Error loading processed contracts: {e}")
        return
    
    contract_types = contracts_df['contract_type'].unique()
    print(f"Contract types in dataset: {contract_types}")
    
    sample_contracts = []
    for contract_type in contract_types:
        type_samples = contracts_df[contracts_df['contract_type'] == contract_type]
        if not type_samples.empty:
            sample_contracts.append(type_samples.iloc[0])
    
    if not sample_contracts:
        if not contracts_df.empty:
            sample_contracts = [contracts_df.iloc[0]]
        else:
            print("No contracts found in the dataset")
            return
    
    sample_contract = sample_contracts[0]
    contract_text = sample_contract['cleaned_text']
    
    print(f"\nApplying Chain of Thought analysis to: {sample_contract['file_path']}")
    print(f"Contract type: {sample_contract['contract_type']}")
    
    if analyzer is None:
        print("Skipping analysis steps because model could not be loaded")
        return
    
    print("\nPerforming Chain of Thought analysis...")
    cot_result = analyzer.chain_of_thought_analysis(contract_text)
    
    if "error" in cot_result:
        print(f"Chain of Thought analysis failed: {cot_result['error']}")
    else:
        with open("cot_analysis.json", "w") as f:
            json.dump(cot_result, f, indent=2)
        
        print("Chain of Thought analysis saved to cot_analysis.json")
    
    clause_types = ["termination", "confidentiality", "intellectual_property"]
    
    for clause_type in clause_types:
        print(f"\nAnalyzing {clause_type} clauses...")
        tot_result = analyze_specific_clause_type(analyzer, contracts_df, clause_type)
        
        if tot_result:
            if "error" in tot_result:
                print(f"Tree of Thought analysis for {clause_type} failed: {tot_result['error']}")
            else:
                with open(f"tot_analysis_{clause_type}.json", "w") as f:
                    json.dump(tot_result, f, indent=2)
                
                print(f"Tree of Thought analysis for {clause_type} saved to tot_analysis_{clause_type}.json")
    
    print("\nPerforming Graph of Thought analysis...")
    got_result = analyzer.graph_of_thought_analysis(contract_text)
    
    if "error" in got_result.get("graph_data", {}):
        print(f"Graph of Thought analysis failed: {got_result['graph_data']['error']}")
    else:
        with open("got_analysis.json", "w") as f:
            json.dump(got_result, f, indent=2)
        
        print("Graph of Thought analysis saved to got_analysis.json")
        
        if "graph_data" in got_result and "error" not in got_result["graph_data"]:
            print("\nVisualizing contract graph...")
            analyzer.visualize_contract_graph(got_result["graph_data"])

if __name__ == "__main__":
    main()