import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama
import faiss
from sentence_transformers import SentenceTransformer
import textwrap
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class LegalContractRAG:
    def __init__(self, model_path, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Initialize the RAG system for legal contracts
        
        Args:
            model_path: Path to the LLM model
            embedding_model_name: Name of the SentenceTransformer model to use
        """
        # Initialize the LLM
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=-1
            )
            print(f"LLM model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            print(f"Embedding model {embedding_model_name} loaded")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.clause_documents = []
    
    def segment_contracts(self, contracts_df, min_chunk_size=200, max_chunk_size=500):
        """
        Segment contracts into clauses and paragraphs for better retrieval
        
        Args:
            contracts_df: DataFrame containing contract data
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
        
        Returns:
            List of document dictionaries with text chunks and metadata
        """
        documents = []
        
        for idx, row in tqdm(contracts_df.iterrows(), total=len(contracts_df), desc="Segmenting contracts"):
            contract_text = row['cleaned_text']
            contract_type = row['contract_type']
            file_path = row['file_path']
            
            # First, try to identify major sections/clauses
            # Common patterns in legal documents
            section_patterns = [
                # Pattern for "1. Section Title"
                r'\n\s*(\d+\.[\s]+[A-Z][^\.]+\.)',
                # Pattern for "Section 1. Section Title"
                r'\n\s*(Section\s+\d+\.?\s+[A-Z][^\.]+\.)',
                # Pattern for "ARTICLE I: TITLE"
                r'\n\s*(ARTICLE\s+[IVX]+\.?:?\s+[A-Z][^\.]+\.)',
                # Pattern for "1.1 Subsection Title"
                r'\n\s*(\d+\.\d+\.?\s+[A-Z][^\.]+\.)'
            ]
            
            # Find all sections
            all_sections = []
            for pattern in section_patterns:
                sections = re.findall(pattern, contract_text)
                all_sections.extend([(s, contract_text.find(s)) for s in sections])
            
            # Sort sections by their position in the document
            all_sections.sort(key=lambda x: x[1])
            
            if len(all_sections) > 1:
                # Extract text between section headers
                for i in range(len(all_sections) - 1):
                    current_section = all_sections[i][0]
                    current_pos = all_sections[i][1]
                    next_pos = all_sections[i+1][1]
                    
                    section_text = contract_text[current_pos:next_pos]
                    
                    # Skip if section is too short
                    if len(section_text) < min_chunk_size:
                        continue
                    
                    documents.append({
                        'text': section_text,
                        'source': file_path,
                        'contract_type': contract_type,
                        'section': current_section.strip(),
                        'doc_type': 'section'
                    })
                
                # Don't forget the last section
                last_section = all_sections[-1][0]
                last_pos = all_sections[-1][1]
                last_section_text = contract_text[last_pos:]
                
                documents.append({
                    'text': last_section_text,
                    'source': file_path,
                    'contract_type': contract_type,
                    'section': last_section.strip(),
                    'doc_type': 'section'
                })
            else:
                # If no clear sections found, split by paragraphs
                paragraphs = re.split(r'\n\s*\n', contract_text)
                
                for i, para in enumerate(paragraphs):
                    # Skip if paragraph is too short
                    if len(para) < min_chunk_size:
                        continue
                    
                    # Further chunk if paragraph is too long
                    if len(para) > max_chunk_size:
                        chunks = textwrap.wrap(para, max_chunk_size)
                        for j, chunk in enumerate(chunks):
                            documents.append({
                                'text': chunk,
                                'source': file_path,
                                'contract_type': contract_type,
                                'section': f"Paragraph {i+1}, Chunk {j+1}",
                                'doc_type': 'paragraph'
                            })
                    else:
                        documents.append({
                            'text': para,
                            'source': file_path,
                            'contract_type': contract_type,
                            'section': f"Paragraph {i+1}",
                            'doc_type': 'paragraph'
                        })
        
        print(f"Created {len(documents)} document chunks from {len(contracts_df)} contracts")
        return documents
    
    def extract_legal_clauses(self, documents):
        """
        Extract specific legal clauses from documents
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            List of clause documents with metadata
        """
        # Common legal clause types to identify
        clause_keywords = {
            "termination": ["terminat", "cancel", "end the agreement"],
            "confidentiality": ["confidential", "non-disclosure", "secret", "proprietary information"],
            "indemnification": ["indemnif", "hold harmless", "defend"],
            "limitation_of_liability": ["limit liability", "no liability", "not be liable"],
            "payment_terms": ["payment", "invoice", "fee", "compensation"],
            "governing_law": ["govern law", "jurisdiction", "venue"],
            "intellectual_property": ["intellectual property", "copyright", "patent", "trademark"],
            "force_majeure": ["force majeure", "act of god", "beyond the control"],
            "warranty": ["warrant", "guarantee", "assurance"],
            "assignment": ["assign", "transfer rights", "delegation"]
        }
        
        clause_documents = []
        
        for doc in tqdm(documents, desc="Extracting legal clauses"):
            text = doc['text'].lower()
            
            for clause_type, keywords in clause_keywords.items():
                # Check if any of the keywords are in the text
                if any(keyword.lower() in text for keyword in keywords):
                    # Create a new document for this clause
                    clause_doc = doc.copy()
                    clause_doc['clause_type'] = clause_type
                    clause_doc['doc_type'] = 'clause'
                    clause_documents.append(clause_doc)
        
        print(f"Extracted {len(clause_documents)} legal clauses")
        return clause_documents
    
    def build_vector_index(self, documents):
        """
        Build a FAISS vector index for fast similarity search
        
        Args:
            documents: List of document dictionaries
        """
        if self.embedding_model is None:
            print("Cannot build index: embedding model not available")
            return
            
        # Store the documents
        self.documents = documents
        
        # Extract the text from each document
        texts = [doc['text'] for doc in documents]
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Normalize the embeddings
        faiss.normalize_L2(embeddings)
        
        # Build the FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
        self.index.add(embeddings)
        
        print(f"Built vector index with {len(documents)} documents")
    
    def search(self, query, k=5):
        """
        Search for documents similar to the query
        
        Args:
            query: The search query
            k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with similarity scores
        """
        if self.embedding_model is None or self.index is None:
            print("Cannot search: embedding model or index not available")
            return []
            
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Normalize the query embedding
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, k)
        
        # Retrieve the documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):  # Valid index
                doc = self.documents[idx].copy()
                doc['similarity'] = float(scores[0][i])
                results.append(doc)
        
        return results
    
    def generate_response(self, query, retrieved_docs, max_context_length=3000):
        """
        Generate a response to the query using the retrieved documents
        
        Args:
            query: The query string
            retrieved_docs: List of retrieved documents
            max_context_length: Maximum context length for the prompt
        
        Returns:
            Generated response
        """
        if self.llm is None:
            return "LLM model not available for generating responses."
            
        # Prepare the context from retrieved documents
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            source = os.path.basename(doc.get('source', 'Unknown'))
            section = doc.get('section', 'N/A')
            doc_text = doc.get('text', '')
            
            # Only add if we have text
            if doc_text:
                doc_context = f"\nDOCUMENT {i}:\nSource: {source}\nSection: {section}\n\n{doc_text[:500]}...\n"
                
                # Check if adding this document would exceed the max context length
                if len(context) + len(doc_context) > max_context_length:
                    break
                    
                context += doc_context
        
        # Prepare a simpler prompt for better stability
        prompt = f"""You are a legal assistant specialized in contract analysis. Use the following contract excerpts to answer the user's question.

CONTRACT EXCERPTS:
{context}

USER QUESTION: {query}

ANSWER:"""
        
        try:
            # Generate the response with modified parameters
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=500,  # Increased max tokens
                temperature=0.2,  # Lower temperature for more reliable output
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            return response['choices'][0]['text'].strip()
        except Exception as e:
            # Better error handling
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def save(self, filepath):
        """
        Save the RAG system
        
        Args:
            filepath: Path to save the system
        """
        # We need to save the documents and rebuild the index later
        # as FAISS indices might not be pickle-compatible
        save_data = {
            'documents': self.documents,
            'clause_documents': self.clause_documents
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"RAG system saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the RAG system
        
        Args:
            filepath: Path to the saved system
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.documents = save_data['documents']
        self.clause_documents = save_data['clause_documents']
        
        # Rebuild the index if we have a valid embedding model
        if self.embedding_model is not None and self.documents:
            texts = [doc['text'] for doc in self.documents]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            faiss.normalize_L2(embeddings)
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
        
        print(f"RAG system loaded from {filepath}")
        print(f"Loaded {len(self.documents)} documents and {len(self.clause_documents)} clause documents")

def load_cuad_labels(dataset_path):
    """
    Load CUAD labels for contract analysis
    
    Args:
        dataset_path: Path to the CUAD dataset
    
    Returns:
        Dictionary mapping contract IDs to labels
    """
    json_path = os.path.join(dataset_path, 'CUAD_v1.json')
    
    if not os.path.exists(json_path):
        print(f"CUAD labels file not found at {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            cuad_data = json.load(f)
        
        # Process the CUAD data
        contract_labels = {}
        
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
                        # Extract label information
                        label_type = question.split(':')[0] if ':' in question else question
                        label_value = [ans.get('text', '') for ans in answers]
                        
                        if title not in contract_labels:
                            contract_labels[title] = {}
                        
                        if label_type not in contract_labels[title]:
                            contract_labels[title][label_type] = []
                        
                        contract_labels[title][label_type].extend(label_value)
        
        print(f"Loaded labels for {len(contract_labels)} contracts")
        return contract_labels
    
    except Exception as e:
        print(f"Error loading CUAD labels: {e}")
        return {}

def identify_contract_types(contracts_df):
    """
    Use TF-IDF to identify common terms and phrases in different contract types
    
    Args:
        contracts_df: DataFrame containing contract data
    
    Returns:
        Dictionary mapping contract types to key terms
    """
    contract_types = contracts_df['contract_type'].unique()
    contract_type_terms = {}
    
    # For each contract type, identify the most distinctive terms
    for contract_type in contract_types:
        # Get contracts of this type
        type_samples = contracts_df[contracts_df['contract_type'] == contract_type]
        
        # Skip if too few samples
        if len(type_samples) < 2:
            contract_type_terms[contract_type] = ["insufficient samples"]
            continue
            
        # Get other contracts
        other_contracts = contracts_df[contracts_df['contract_type'] != contract_type]
        
        # Create a corpus for TF-IDF
        type_corpus = type_samples['cleaned_text'].tolist()
        other_corpus = other_contracts['cleaned_text'].sample(min(len(other_contracts), len(type_corpus))).tolist()
        
        # Create labels
        corpus = type_corpus + other_corpus
        labels = [1] * len(type_corpus) + [0] * len(other_corpus)
        
        # Calculate TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(corpus)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF for each class
        type_tfidf = X[:len(type_corpus)].mean(axis=0)
        other_tfidf = X[len(type_corpus):].mean(axis=0)
        
        # Calculate the difference in TF-IDF between the contract type and others
        tfidf_diff = np.array(type_tfidf - other_tfidf)[0]
        
        # Get the top terms
        top_indices = tfidf_diff.argsort()[-20:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        contract_type_terms[contract_type] = top_terms
    
    return contract_type_terms

def evaluate_contract_classification(rag, contracts_df, num_samples=5):
    """
    Evaluate the system's ability to classify contract types
    
    Args:
        rag: LegalContractRAG instance
        contracts_df: DataFrame containing contract data
        num_samples: Number of contracts to sample for evaluation
    """
    if rag.llm is None:
        print("Cannot evaluate: LLM model not available")
        return
        
    # Sample contracts for evaluation
    sample_contracts = contracts_df.sample(min(num_samples, len(contracts_df)))
    
    correct = 0
    
    for idx, row in sample_contracts.iterrows():
        contract_text = row['cleaned_text'][:2000]  # Limit to 2000 chars for faster processing
        contract_type = row['contract_type']
        
        # Query to determine contract type
        query = "What type of contract is this? Answer with a single word or phrase."
        
        # Create test document
        test_doc = {
            'text': contract_text,
            'source': os.path.basename(row['file_path']),
            'section': 'Sample for classification'
        }
        
        print(f"\nAnalyzing contract: {os.path.basename(row['file_path'])}")
        print(f"Actual contract type: {contract_type}")
        
        try:
            # Generate response with increased max_tokens and simplified prompt
            response = rag.llm.create_completion(
                prompt=f"CONTRACT TEXT:\n{contract_text[:1000]}\n\nQUESTION: What type of contract is this? Answer with a single phrase.\nANSWER:",
                max_tokens=50,  # Shorter response
                temperature=0.1,  # More deterministic
                stop=["</s>", "\n\n"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            print(f"Generated response: {generated_text}")
            
            # Check if the contract type is in the response (case insensitive)
            if contract_type.lower() in generated_text.lower():
                correct += 1
                print("✓ Correct classification")
            else:
                print("✗ Incorrect classification")
                
        except Exception as e:
            print(f"Error generating response: {e}")
            print("✗ Error in classification")
    
    accuracy = correct / num_samples if num_samples > 0 else 0
    print(f"\nClassification accuracy: {accuracy:.2%} ({correct}/{num_samples})")

def generate_example_responses(rag, contracts_df):
    """
    Generate example responses to test the RAG system
    """
    if len(contracts_df) == 0:
        print("No contracts to analyze")
        return
        
    # Example queries for legal contract understanding
    example_queries = [
        "What are the termination conditions in this contract?",
        "Explain the confidentiality provisions in this agreement",
        "Summarize the intellectual property rights in this contract"
    ]
    
    # Take a sample contract
    sample_contract = contracts_df.iloc[0]
    contract_text = sample_contract['cleaned_text']
    contract_file = os.path.basename(sample_contract['file_path'])
    
    print(f"\nUsing contract: {contract_file}")
    
    # Try the example queries
    for query in example_queries:
        print(f"\nQuery: {query}")
        
        try:
            # Direct model query for testing
            prompt = f"""
            You are a legal assistant analyzing contracts.
            
            CONTRACT TEXT (excerpt):
            {contract_text[:1500]}
            
            USER QUESTION: {query}
            
            Provide a clear, concise answer based on the contract excerpt.
            ANSWER:
            """
            
            # Generate response directly using the LLM
            response = rag.llm.create_completion(
                prompt=prompt,
                max_tokens=300,
                temperature=0.2,
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            print("Direct model response:")
            print(response['choices'][0]['text'])
            
            # Now try the RAG approach
            print("\nRAG system response (might be empty if there are issues):")
            # Retrieve relevant documents
            retrieved_docs = rag.search(query, k=2)
            
            # Add the sample contract text to provide context
            context_doc = {
                'text': contract_text[:1500],
                'source': contract_file,
                'section': 'Context'
            }
            
            # Try to generate response with RAG
            try:
                rag_response = rag.generate_response(query, retrieved_docs + [context_doc])
                print(rag_response)
            except Exception as e:
                print(f"Error with RAG response: {e}")
                
        except Exception as e:
            print(f"Error generating response: {e}")

def main():
    # Path to the model
    model_path = "./Mistral-7B-Instruct-v0.3.Q5_K_M.gguf"
    
    # Path to CUAD dataset
    cuad_dataset_path = "./CUAD_v1"  # Update this path to match your directory
    
    # Load processed contracts
    try:
        contracts_df = pd.read_pickle("processed_contracts.pkl")
        print(f"Loaded {len(contracts_df)} processed contracts")
    except FileNotFoundError:
        print("Error: Run Phase 1 script first to generate processed_contracts.pkl")
        return
    
    # Load CUAD labels
    cuad_labels = load_cuad_labels(cuad_dataset_path)
    
    try:
        # Initialize the RAG system
        rag = LegalContractRAG(model_path)
        
        # Check if RAG system already exists
        if os.path.exists("legal_rag.pkl"):
            print("Loading existing RAG system...")
            rag.load("legal_rag.pkl")
        else:
            print("Building new RAG system...")
            # Segment contracts into chunks
            documents = rag.segment_contracts(contracts_df)
            
            # Extract legal clauses
            rag.clause_documents = rag.extract_legal_clauses(documents)
            
            # Build vector index
            rag.build_vector_index(documents)
            
            # Save the RAG system
            rag.save("legal_rag.pkl")
        
        # Evaluate contract classification
        print("\nEvaluating contract classification...")
        evaluate_contract_classification(rag, contracts_df, num_samples=3)
        
        # Generate example responses
        print("\nTesting system with example queries...")
        generate_example_responses(rag, contracts_df)
        
        # Interactive query loop (only run if directly executed)
        if __name__ == "__main__":
            print("\nEntering interactive query mode. Type 'exit' to quit.")
            
            # Choose a sample contract for context
            sample_contract = contracts_df.iloc[0]
            contract_text = sample_contract['cleaned_text']
            
            while True:
                query = input("\nEnter your legal contract question (or 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                
                try:
                    # Retrieve relevant documents
                    retrieved_docs = rag.search(query, k=3)
                    
                    # Add the sample contract for context
                    context_doc = {
                        'text': contract_text[:2000],
                        'source': os.path.basename(sample_contract['file_path']),
                        'section': 'Context'
                    }
                    
                    # Print the retrieved documents
                    print(f"\nRetrieved {len(retrieved_docs)} documents:")
                    for i, doc in enumerate(retrieved_docs, 1):
                        source = os.path.basename(doc.get('source', 'Unknown'))
                        section = doc.get('section', 'N/A')
                        similarity = doc.get('similarity', 0)
                        print(f"{i}. {source} - {section} (similarity: {similarity:.2f})")
                    
                    # Generate response
                    response = rag.generate_response(query, retrieved_docs + [context_doc])
                    
                    print("\nResponse:")
                    print(response)
                    
                except Exception as e:
                    print(f"Error processing query: {e}")
                    # Fall back to direct model query
                    try:
                        prompt = f"CONTRACT TEXT:\n{contract_text[:1500]}\n\nQUESTION: {query}\n\nANSWER:"
                        response = rag.llm.create_completion(
                            prompt=prompt,
                            max_tokens=300,
                            temperature=0.2,
                            stop=["</s>", "\n\n"],
                            echo=False
                        )
                        print("\nDirect model response:")
                        print(response['choices'][0]['text'])
                    except Exception as e2:
                        print(f"Error with direct query: {e2}")
    
    except Exception as e:
        print(f"Error initializing or running the RAG system: {e}")
        print("Please check if the model file exists and is accessible.")

if __name__ == "__main__":
    main()