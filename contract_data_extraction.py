import os
import re
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from PyPDF2 import PdfReader
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def clean_legal_text(text):
    """
    Clean and preprocess legal text
    """
    # Remove page numbers (common in legal documents)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Remove headers/footers (simplified - may need customization)
    text = re.sub(r'\n\s*.{1,50}\s*\n\s*Page \d+ of \d+\s*\n', '\n', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove special characters but keep legal symbols
    text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\{\}\$\%\&\/\'\"\-\_\+\=\*\#\@\!\?§]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_legal_text(text):
    """
    Tokenize legal text into sentences and perform basic NLP analysis
    """
    # Split into sentences
    sentences = sent_tokenize(text)
    
    # Process with spaCy
    docs = list(nlp.pipe(sentences, disable=["ner"]))
    
    return docs

def analyze_pos_distribution(docs):
    """
    Analyze POS tag distribution in the documents
    """
    pos_counts = Counter()
    
    for doc in docs:
        for token in doc:
            pos_counts[token.pos_] += 1
    
    return pos_counts

def extract_legal_entities(text):
    """
    Extract legal entities using spaCy's NER
    """
    doc = nlp(text)
    entities = {
        'ORG': [],  # Organizations (parties)
        'PERSON': [], # People
        'DATE': [],  # Dates
        'MONEY': [], # Monetary values
        'LAW': [],   # Laws/regulations
    }
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    
    return entities

def train_word_embeddings(docs, embedding_dim=100):
    """
    Train Word2Vec embeddings on legal documents
    """
    # Prepare sentences for Word2Vec
    sentences = [[token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
                for doc in docs]
    
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=5, workers=4)
    
    return model

def visualize_embeddings(model, legal_terms):
    """
    Visualize word embeddings for specific legal terms
    """
    # Filter for terms that exist in the vocabulary
    legal_terms = [term for term in legal_terms if term in model.wv.key_to_index]
    
    if len(legal_terms) == 0:
        print("None of the specified legal terms found in vocabulary")
        return None
    
    # Get embeddings for legal terms
    embeddings = np.array([model.wv[term] for term in legal_terms])
    
    # Adjust perplexity based on number of samples
    perplexity = min(5, len(legal_terms) - 1) if len(legal_terms) > 1 else 1
    
    # Apply t-SNE for dimensionality reduction
    if len(legal_terms) >= 3:  # Need at least 3 points for t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        # For fewer points, just use PCA or first 2 dimensions
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'term': legal_terms,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y')
    
    # Add term labels
    for i, row in df.iterrows():
        plt.annotate(row['term'], (row['x'], row['y']))
    
    plt.title("Visualization of Legal Term Embeddings")
    plt.savefig("legal_embeddings.png")
    plt.close()
    
    return df

def extract_contract_type_from_filename(filename):
    """
    Extract contract type from a filename
    """
    contract_type_keywords = [
        "Affiliate", "Agreement", "Agency", "Collaboration", "Consulting", 
        "Development", "Distributor", "Endorsement", "Franchise", "Hosting",
        "IP", "Joint_Venture", "License", "Maintenance", "Manufacturing",
        "Marketing", "Non_Compete", "Outsourcing", "Promotion", "Reseller",
        "Service", "Sponsorship", "Strategic", "Supply", "Transportation"
    ]
    
    # Convert to uppercase for better matching
    upper_filename = filename.upper()
    
    # Check for common contract type indicators
    for keyword in contract_type_keywords:
        if keyword.upper() in upper_filename or keyword.upper().replace('_', ' ') in upper_filename:
            return keyword
    
    # Check for special cases
    if "CONTENT LICENSE" in upper_filename or "LICENSE AGREEMENT" in upper_filename:
        return "License"
    if "TRADEMARK" in upper_filename and "LICENSE" in upper_filename:
        return "Trademark_License"
    if "JOINT FILING" in upper_filename:
        return "Joint_Filing"
    
    return "Unknown"

def process_contract_dataset(dataset_path, use_txt=True, sample_size=None):
    """
    Process the CUAD dataset
    
    Args:
        dataset_path: Path to the CUAD dataset
        use_txt: If True, use the .txt files in full_contract_txt instead of PDFs
        sample_size: Number of contracts to sample per type (None for all)
    
    Returns:
        DataFrame with contract data
    """
    all_texts = []
    contract_types = []
    file_paths = []
    
    if use_txt:
        # Use pre-extracted text files instead of PDFs (recommended)
        txt_folder = os.path.join(dataset_path, 'full_contract_txt')
        
        if not os.path.exists(txt_folder):
            print(f"Text folder {txt_folder} not found. Falling back to PDF extraction.")
            use_txt = False
        else:
            txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
            
            if sample_size:
                # Group files by contract type first
                contract_type_files = {}
                for txt_file in txt_files:
                    # Extract contract type from filename
                    contract_type = extract_contract_type_from_filename(txt_file)
                    
                    if contract_type not in contract_type_files:
                        contract_type_files[contract_type] = []
                    contract_type_files[contract_type].append(txt_file)
                
                # Sample from each contract type
                sampled_files = []
                for contract_type, files in contract_type_files.items():
                    sampled_files.extend(files[:sample_size])
                txt_files = sampled_files
            
            for txt_file in tqdm(txt_files, desc="Processing text files"):
                txt_path = os.path.join(txt_folder, txt_file)
                
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Determine contract type from filename
                    contract_type = extract_contract_type_from_filename(txt_file)
                    
                    all_texts.append(text)
                    contract_types.append(contract_type)
                    file_paths.append(txt_path)
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
    
    if not use_txt or not all_texts:
        # Fall back to PDF extraction if needed
        print("Using PDF extraction...")
        pdf_folder = os.path.join(dataset_path, 'full_contract_pdf')
        
        if not os.path.exists(pdf_folder):
            raise ValueError(f"Neither text nor PDF folders found in {dataset_path}")
        
        # Process each part (Part_I, Part_II, etc.)
        for part_dir in os.listdir(pdf_folder):
            part_path = os.path.join(pdf_folder, part_dir)
            if not os.path.isdir(part_path):
                continue
            
            # Process each contract type directory
            for contract_type_dir in os.listdir(part_path):
                contract_type_path = os.path.join(part_path, contract_type_dir)
                if not os.path.isdir(contract_type_path):
                    continue
                
                # List PDF files
                pdf_files = [f for f in os.listdir(contract_type_path) if f.lower().endswith('.pdf')]
                
                if sample_size:
                    pdf_files = pdf_files[:sample_size]
                
                for pdf_file in tqdm(pdf_files, desc=f"Processing {contract_type_dir} contracts"):
                    pdf_path = os.path.join(contract_type_path, pdf_file)
                    
                    # Extract text
                    raw_text = extract_text_from_pdf(pdf_path)
                    
                    if raw_text:
                        all_texts.append(raw_text)
                        contract_types.append(contract_type_dir)
                        file_paths.append(pdf_path)
    
    # Create a DataFrame to organize the data
    contracts_df = pd.DataFrame({
        'contract_type': contract_types,
        'file_path': file_paths,
        'raw_text': all_texts
    })
    
    # Clean texts
    contracts_df['cleaned_text'] = contracts_df['raw_text'].apply(clean_legal_text)
    
    return contracts_df

def main():
    # Set the path to your CUAD dataset
    cuad_dataset_path = "./CUAD_v1"  # Update this path to match your directory
    
    # Process a sample of the dataset (to speed up initial development)
    # Use txt files by default for faster processing
    print("Processing CUAD dataset...")
    contracts_df = process_contract_dataset(cuad_dataset_path, use_txt=True, sample_size=3)
    
    print(f"Processed {len(contracts_df)} contracts")
    print(f"Contract types: {contracts_df['contract_type'].value_counts().to_dict()}")
    
    # Save the processed data
    contracts_df.to_pickle("processed_contracts.pkl")
    print("Saved processed data to processed_contracts.pkl")
    
    # Analyze a single contract for demonstration
    sample_contract = contracts_df.iloc[0]
    print(f"\nAnalyzing sample contract: {sample_contract['file_path']}")
    print(f"Contract type: {sample_contract['contract_type']}")
    
    # Tokenize and analyze
    docs = tokenize_legal_text(sample_contract['cleaned_text'])
    
    # Analyze POS distribution
    pos_counts = analyze_pos_distribution(docs)
    print("\nPOS Distribution:")
    for pos, count in pos_counts.most_common():
        print(f"{pos}: {count}")
    
    # Extract entities
    entities = extract_legal_entities(sample_contract['cleaned_text'][:10000])  # Limit to first 10k chars for speed
    print("\nExtracted Entities:")
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}: {entity_list[:5]}")  # Show first 5 of each type
    
    # Train Word2Vec on all documents
    print("\nTraining Word2Vec model...")
    all_docs = []
    for text in tqdm(contracts_df['cleaned_text']):
        all_docs.extend(tokenize_legal_text(text))
    
    word2vec_model = train_word_embeddings(all_docs)
    word2vec_model.save("legal_word2vec.model")
    print("Saved Word2Vec model to legal_word2vec.model")
    
    # Visualize legal term embeddings
    legal_terms = [
        "contract", "agreement", "party", "parties", "clause",
        "term", "provision", "breach", "liability", "indemnify",
        "termination", "confidential", "dispute", "payment", "warranty"
    ]
    
    # Filter for terms that exist in the vocabulary
    available_terms = [term for term in legal_terms if term in word2vec_model.wv.key_to_index]
    print(f"\nFound {len(available_terms)} of {len(legal_terms)} legal terms in vocabulary:")
    print(", ".join(available_terms))
    
    visualization_df = visualize_embeddings(word2vec_model, legal_terms)
    if visualization_df is not None:
        print("Embedding visualization saved as 'legal_embeddings.png'")
    
    # Analyze legal clauses
    print("\nAnalyzing common legal clauses...")
    clause_terms = {
        "termination": ["terminat", "cancel", "end the agreement"],
        "confidentiality": ["confidential", "non-disclosure"],
        "indemnification": ["indemnif", "hold harmless"],
        "liability": ["limit liability", "no liability"],
        "intellectual_property": ["intellectual property", "copyright", "patent"]
    }
    
    for clause, terms in clause_terms.items():
        similar_terms = []
        for term in terms:
            if term in word2vec_model.wv:
                similar = word2vec_model.wv.most_similar(term, topn=5)
                similar_terms.extend([word for word, _ in similar])
        
        if similar_terms:
            print(f"{clause.capitalize()} related terms: {', '.join(similar_terms)}")
        else:
            print(f"No related terms found for {clause}")
            
    # Print some examples of word embeddings
    print("\nExample word similarities:")
    try:
        similarity = word2vec_model.wv.similarity('agreement', 'contract')
        print(f"Similarity between 'agreement' and 'contract': {similarity:.4f}")
    except KeyError:
        print("Terms 'agreement' and 'contract' not both in vocabulary")
    
    try:
        similarity = word2vec_model.wv.similarity('termination', 'breach')
        print(f"Similarity between 'termination' and 'breach': {similarity:.4f}")
    except KeyError:
        print("Terms 'termination' and 'breach' not both in vocabulary")

if __name__ == "__main__":
    main()
