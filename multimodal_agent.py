import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_cpp import Llama
import speech_recognition as sr
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import cv2
from pdf2image import convert_from_path
import fastapi
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn
from io import BytesIO

class MultimodalLegalAgent:
    def __init__(self, model_path, ocr_config=None, device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Using device: {self.device}")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_batch=512,
                #n_gpu_layers=-1 if self.device == "cuda" else 0
                n_gpu_layers=0
            )
            print(f"LLM loaded: {model_path}")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None
            
        self.ocr_config = ocr_config or '--psm 3 --oem 3'
        self.recognizer = sr.Recognizer()
        try:
            self.layout_model = AutoModelForImageClassification.from_pretrained(
                "microsoft/layoutlm-base-uncased"
            ).to(self.device)
            self.layout_feature_extractor = AutoFeatureExtractor.from_pretrained(
                "microsoft/layoutlm-base-uncased"
            )
            print("Document layout model loaded")
        except Exception as e:
            print(f"Warning: Could not load layout model: {e}")
            self.layout_model = None
            self.layout_feature_extractor = None
        
        try:
            self.rag_system = self._load_rag_system()
            print("RAG system loaded")
        except Exception as e:
            print(f"Warning: Could not load RAG system: {e}")
            self.rag_system = None
    
    def _load_rag_system(self):
        from rag_system import LegalContractRAG
        
        rag_path = "legal_rag.pkl"
        model_path = "./Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
        
        if os.path.exists(rag_path):
            rag = LegalContractRAG(model_path)
            rag.load(rag_path)
            return rag
        else:
            print(f"Warning: RAG system file not found at {rag_path}")
            return None
    
    def process_text_input(self, text, query=None):
        print("="*50)
        print(f"Received text input (first 200 chars): {text[:200]}...")
        print(f"Sending prompt to model with length: {len(text)} characters")
        print(f"Received query: {query}")
        
        if self.llm is None:
            return {"error": "LLM not initialized"}
        max_safe_length = 2048
        max_chunk_length = 1500
        
        if len(text) > max_safe_length:
            print(f"Text is too long ({len(text)} chars), applying chunking strategy")
            
            chunks = []
            paragraphs = text.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < max_chunk_length:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            print(f"Split text into {len(chunks)} chunks")
            
            selected_chunks = []
            
            if query:
                query_terms = query.lower().split()
                
                chunk_scores = []
                for i, chunk in enumerate(chunks):
                    position_score = 1.0 if i == 0 or i == len(chunks) - 1 else 0.5
                    
                    term_score = 0
                    chunk_lower = chunk.lower()
                    for term in query_terms:
                        if term in chunk_lower:
                            term_score += 1
                    
                    total_score = position_score + (term_score * 2)
                    chunk_scores.append((i, total_score))
                
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                selected_indices = [idx for idx, score in chunk_scores[:3]]
                selected_indices.sort()
                
                selected_chunks = [chunks[idx] for idx in selected_indices]
            else:
                selected_chunks = [chunks[0]]
                if len(chunks) > 2:
                    selected_chunks.append(chunks[len(chunks) // 2])
                if len(chunks) > 1:
                    selected_chunks.append(chunks[-1])
            
            print(f"Selected {len(selected_chunks)} most relevant chunks")
            
            marked_chunks = []
            for i, chunk in enumerate(selected_chunks):
                if i == 0:
                    marked_chunks.append(f"[BEGINNING OF CONTRACT]\n{chunk}")
                elif i == len(selected_chunks) - 1:
                    marked_chunks.append(f"[LATER PART OF CONTRACT]\n{chunk}")
                else:
                    marked_chunks.append(f"[MIDDLE SECTION OF CONTRACT]\n{chunk}")
            
            analysis_text = "\n\n[...]\n\n".join(marked_chunks)
        else:
            analysis_text = text
        
        if query is None:
            # General analysis without specific query
            prompt = f"""
            You are a legal assistant analyzing a contract. Review the following contract text and provide a concise analysis highlighting the key terms, obligations, and potential issues.
            
            CONTRACT TEXT:
            {analysis_text}
            
            Note: This may be a partial extract of a longer contract.
            
            YOUR ANALYSIS:
            """
        else:
            # With a specific query
            prompt = f"""
            You are a legal assistant analyzing a contract. The following text was provided by a user. 
            
            CONTRACT TEXT:
            {analysis_text}
            
            Note: This may be a partial extract of a longer contract.
            
            USER QUESTION: {query}
            
            YOUR ANSWER (based on the provided text):
            """
        
        print(f"Final prompt length: {len(prompt)} characters")
        print(f"Prompt being sent to model (first 500 chars): {prompt[:500]}...")
        
        try:
            response = self.llm.create_completion(
                prompt=prompt,
                max_tokens=512,  # Reduced to save memory
                temperature=0.2,
                stop=["</s>", "\n\n\n"],
                echo=False
            )
            
            response_text = response['choices'][0]['text'].strip()
            print(f"Model response (first 200 chars): {response_text[:200]}...")
            
            return response_text
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"Error: {error_msg}")
            return {"error": error_msg}
    
    def process_voice_input(self, audio_file=None, microphone=False, duration=5):
        text = ""
        
        try:
            if microphone:
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source, timeout=duration)
            elif audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)
            else:
                return {"error": "No audio source provided"}
            
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            
            return {"transcribed_text": text}
        except sr.UnknownValueError:
            return {"error": "Speech Recognition could not understand audio"}
        except sr.RequestError as e:
            return {"error": f"Could not request results from Speech Recognition service: {e}"}
        except Exception as e:
            return {"error": f"Error processing voice input: {str(e)}"}
    
    def search_knowledge_base(self, query):
        if self.rag_system is None:
            return {"error": "RAG system not initialized"}
        
        try:
            retrieved_docs = self.rag_system.search(query, k=3)
            response = self.rag_system.generate_response(query, retrieved_docs)
            
            return {
                "result": response,
                "retrieved_docs": [
                    {
                        "source": doc.get("source", "Unknown"),
                        "section": doc.get("section", "Unknown"),
                        "snippet": doc.get("text", "")[:200] + "..." if doc.get("text") else ""
                    }
                    for doc in retrieved_docs
                ]
            }
        except Exception as e:
            error_msg = f"Error searching knowledge base: {str(e)}"
            print(f"Error: {error_msg}")
            return {"error": error_msg}

    def process_image_input(self, image_path=None, image_data=None, query=None):
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path)
            elif image_data:
                image = Image.open(BytesIO(image_data))
            else:
                return {"error": "No valid image provided"}
            
            img_gray = image.convert('L')
            threshold = 150
            img_binary = img_gray.point(lambda x: 0 if x < threshold else 255, '1')
            extracted_text = pytesseract.image_to_string(img_binary, config=self.ocr_config)
            document_structure = {}
            if self.layout_model and self.layout_feature_extractor:
                image_np = np.array(image)
                try:
                    inputs = self.layout_feature_extractor(images=image_np, return_tensors="pt").to(self.device)
                    outputs = self.layout_model(**inputs)
                    predicted_class = outputs.logits.argmax(-1).item()
                    
                    layout_classes = ["text", "title", "list", "table", "figure"]
                    document_structure["layout_type"] = layout_classes[predicted_class] if predicted_class < len(layout_classes) else "unknown"
                except Exception as e:
                    print(f"Layout analysis failed: {e}")
                    document_structure["layout_type"] = "unknown"
            
            if query and extracted_text:
                analysis = self.process_text_input(extracted_text, query)
            else:
                analysis = {"message": "Text extracted but no query provided for analysis"}
            
            return {
                "extracted_text": extracted_text,
                "document_structure": document_structure,
                "analysis": analysis
            }
        
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def process_pdf_input(self, pdf_path, query=None, page_ranges=None):
        try:
            if not os.path.exists(pdf_path):
                return {"error": f"PDF file not found: {pdf_path}"}
            
            pages = []
            if page_ranges:
                ranges = page_ranges.split(',')
                for r in ranges:
                    if '-' in r:
                        start, end = map(int, r.split('-'))
                        pages.extend(range(start, end + 1))
                    else:
                        pages.append(int(r))
            
            pdf_images = convert_from_path(
                pdf_path, 
                dpi=300,
                first_page=pages[0] if pages else None,
                last_page=pages[-1] if pages else None
            )
            
            all_text = ""
            for i, img in enumerate(pdf_images):
                page_num = pages[i] if pages else i + 1
                print(f"Processing page {page_num}...")
                
                result = self.process_image_input(image_data=BytesIO(), query=None)
                if "error" not in result:
                    all_text += f"\n\n--- Page {page_num} ---\n\n"
                    all_text += result["extracted_text"]
            
            if query and all_text:
                analysis = self.process_text_input(all_text, query)
            else:
                analysis = {"message": "Text extracted but no query provided for analysis"}
            
            return {
                "extracted_text": all_text,
                "total_pages": len(pdf_images),
                "analysis": analysis
            }
        
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}


class TextQuery(BaseModel):
    text: str
    query: Optional[str] = None

class VoiceQuery(BaseModel):
    duration: Optional[int] = 5

class ImageQuery(BaseModel):
    query: Optional[str] = None

class PDFQuery(BaseModel):
    query: Optional[str] = None
    page_ranges: Optional[str] = None

app = FastAPI(title="Multimodal Legal Contract Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    model_path = "./Mistral-7B-Instruct-v0.3.Q5_K_M.gguf"
    app.state.agent = MultimodalLegalAgent(model_path)

@app.post("/analyze/text")
async def analyze_text(data: TextQuery):
    agent = app.state.agent
    # Add debug print here
    print(f"API received text input (length: {len(data.text)}) and query: {data.query}")
    result = agent.process_text_input(data.text, data.query)
    return {"result": result}

@app.post("/analyze/voice")
async def analyze_voice(audio_file: UploadFile = File(...)):
    agent = app.state.agent
    with open("temp_audio.wav", "wb") as f:
        f.write(await audio_file.read())
    
    result = agent.process_voice_input(audio_file="temp_audio.wav")
    
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    
    return result

@app.post("/record/voice")
async def record_voice(data: VoiceQuery):
    agent = app.state.agent
    result = agent.process_voice_input(microphone=True, duration=data.duration)
    
    if "transcribed_text" in result:
        query = result["transcribed_text"]
        analysis = agent.process_text_input("", query)
        result["analysis"] = analysis
    
    return result

@app.post("/search/knowledge_base")
async def search_knowledge_base(query: str = Form(...)):
    agent = app.state.agent
    result = agent.search_knowledge_base(query)
    return result

@app.post("/analyze/image")
async def analyze_image(
    query: str = Form(None),
    image_file: UploadFile = File(...)
):
    agent = app.state.agent
    image_data = await image_file.read()
    
    result = agent.process_image_input(image_data=image_data, query=query)
    return result

@app.post("/analyze/pdf")
async def analyze_pdf(
    query: str = Form(None),
    page_ranges: str = Form(None),
    pdf_file: UploadFile = File(...)
):
    agent = app.state.agent
    with open("temp_pdf.pdf", "wb") as f:
        f.write(await pdf_file.read())
    
    result = agent.process_pdf_input("temp_pdf.pdf", query, page_ranges)
    
    if os.path.exists("temp_pdf.pdf"):
        os.remove("temp_pdf.pdf")
    
    return result

if __name__ == "__main__":
    uvicorn.run("multimodal_agent:app", host="0.0.0.0", port=8000, reload=True)
