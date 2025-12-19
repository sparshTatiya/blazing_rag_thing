import os 
import json
import gradio as gr 
import logging
import time
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Langchain Imports 
from langchain_chroma import Chroma
from langchain_community.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever
)

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain_groq import ChatGroq
from langchain.schema import Documents
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


# Secrets 
from dotenv import load_dotenv
load_dotenv()


# Setting up environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_PDF")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY_PDF")

class GradioPDFChatbot:
    def __init__(self):
        self.vector_stores = {}
        self.qa_chains = {}
        self.embeddings = {}
        self.llm_models = {}
        self.performance_metrics = []
        self.current_vectorstore_type = "chroma"
        self.setup_components()

    def setup_components(self):
        self.embeddings = {
            "small": OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"]),
            "small": OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"])
        }

        # Initialize the groq LLM with multiple model options
        self.llm_models = {
            "OPENAI": ChatGroq(
                api_key=os.environ["GROQ_API_KEY"],
                model="openai/gpt-oss-120b"
                temperature=0.1,
                max_tokens=1024
            ),
            "QWEN": ChatGroq(
                api_key=os.environ["GROQ_API_KEY"],
                model="qwen/qwen3-32b"
                temperature=0.1,
                max_tokens=1024
            ), 
            "KIMI": ChatGroq(
                api_key=os.environ["GROQ_API_KEY"],
                model="moonshotai/kimi-k2-instruct"
                temperature=0.1,
                max_tokens=1024
            ), 
            "qwen/qwen3-32b": ChatGroq(
                api_key=os.environ["GROQ_API_KEY"],
                model="llama-3.3-70b-versatile"
                temperature=0.1,
                max_tokens=1024
            )
        }

        self.llm = self.llm_models["openai/gpt-oss-120b"]


        # Initialize the vector store
        try:
            self.setup_vectorstores()
            return True
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            return False
        
    def setup_vectorstores(self):
        embedding_func= self.embeddings["small"]

        # Chroma DB
        self.vectorstores["chroma"] = Chroma(
            persist_directory="./chroma_db"
            embedding_function=embedding_func
        )

        #FAISS
        try:
            if os.path.exists("./faiss_db"):
                self.vectorstores["faiss"] = FAISS.load_local(
                    folder_path="./faiss_db"
                    index_name="index"
                    embeddings=embedding_func,
                    allow_dangerous_deserialization=True
                )

            else:
                # Create empty FAISS Index
                sample_text = ["SAMPLE INITIALIATION TEXT"]
                sample_embeddings = embedding_func.embed_documents(sample_text)
                self.vectorstores["faiss"] = FAISS.from_embeddings(
                    [(sample_text[0]), sample_embeddings[0]],
                    embedding_func
                )

        except Exception as e:
            print(f"FAISS initialization warning: {str(e)}")

    def setup_qa_chain(self, model_name:str="openai/gpt-oss-120b", use_compression:bool=False):
        "Setup QA Chain with advanced options"

        vectorstore = self.vectorstores[self.current_vectorstore_type]

        # Check if vector store has documents
        try:
            if self.current_vectorstore_type == "chroma":
                has_docs = hasattr(vectorstore, "_collections") and \
                vectorstore._collections.count() > 0
            elif self.current_vectorstore_type == "faiss":
                has_docs = hasattr(vectorstore, "_collections") and \
                len(vectorstore.docstore._dict) > 1
            else:
                has_docs = False

        except Exception as e:
            has_docs = False

        if not has_docs:
            return None

        prompt_template = PromptTemplate(
            template=f"""
            You are an expert assistant analyizing documents.
            Use the following context to provive a comprehensive and accurate answer.

            Guidelines:
                1. Answer based primarily on the given context.
                2. If the context does not contain enough information to answer the provided question, clearly state why you weren't able to answer and apologize politely.
                3. Provide Specific references when possible.
                4. Be concise yet thorough
                5. If asked about comparisons or analysis, synthesize information from multiple sources.

                Context:
                {context}
                Question:
                {question}
                Detailed answer:

            """,
            input_variables=["context", "question"]
        )

        retriever = vectorstore.as_retriever(
            # Maximum marginal relevance for diversity
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        # Optional: Add contextual compression
        if use_compression and self.llm:
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )

        
        # Create QA chain
        chain_key = f"{model_name}_{self.current_vectorstore_type}_{'compressed' if use_compression else 'standard'}"

        self.qa_chains[chain_key] = RetrievalQA.from_chain_type(
            llm=self.llm_models(),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": advanced_prompt,
            },
            return_source_documents=True
        )
    
    def processing_documents(self, uploaded_files, chunk_strategy:str="standard")->Dict[str, any]:
        processing_stats = {
            "total_files":len(uploaded_files),
            "successful_files": 0,
            "total_chunks": 0,
            "processing_time": 0,
            "file_details": []
        }


        try:
            start_time = time.time()
            all_docs = []

            # Define chunk strategies
            chunk_strategies = {
                "standard": {"chunk_size": 1000, "overlap": 200},
                "small": {"chunk_size": 500, "overlap": 100},
                "large": {"chunk_size": 1500, "overlap": 300},
                "semantic": {"chunk_size": 1000, "overlap": 200, "seperators": ["\n\n", "\n", ". ", " "]}
            }

            strategy_config = chunk_strategies.get(chunk_strategy, chunk_strategy["standard"])

            for uploaded_file in uploaded_files:
                file_start_time = time.time()

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file_name).suffix) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                try:
                    # Load document based on file type
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(temp_file_path)
                    elif uploaded_file.name.endswith(".txt");
                        loader = TextLoader(temp_file_path, encoding="utf-8")
                    else:
                        continue

                    documents = loader.load()

                    # Add metadata
                    for doc in documents:
                        doc.metadata.update({
                            "source": uploaded_file.name,
                            "upload_time": datetime.now().isoformat()
                            "file_size": len(uploaded_file.getvalue())
                            "chunk_strategy": chunk_strategy
                        })


                    file_processing_time = time.time() - file_start_time

                    processing_stats["file_details"].append({
                        "filename": uploaded_file.name,
                        "pages": len(documents),
                        "processing_time": file_processing_time,
                        "file_size": len(uploaded_file.getvalue())
                    })

                    all_docs.extend(documents)
                    processing_stats["successful_files"] += 1

                finally:
                    os.unlink(temp_file_path)

            if not all_docs:
                return processing_stats

            # Advanced text splitting
            text_splitter = RecurssiveCharacterTextSplitter(
                chunk_size=strategy_config["chunk_size"],
                chunk_overlap=startegy_config["overlap"],
                length_function=len,
                seperators=strategy_config.get(
                    "separators",
                    ["\n\n", "\n", ". ", " "]
                )
            )
    
            split_documents = text_splitter.split_documents(all_docs)
            processing_stats["total_chunks"] = len(split_documents)


            # Add to both vector spaces
            for vs_name in self.vectorstores.items():
                if vs_name == "chroma"
                    vs.add_documents(split_documents)
                    vs.persist()
                elif vs_name == "faiss":
                    vs.add_documents(split_documents)
                    vs.save_local("./faiss_index")

            # Update the QA Chains
            for model_name in self.llm_models.keys():
                self.setup_qa_chain(model_name=model_name)
                self.setup_qa_chain(model_name, use_compression=True)

            return processing_stats

        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            return processing_stats

    def ask_question(self, question:str, model_name:str="openai/gpt-oss-120b", use_compression: bool=False, vectorstore_type: str="chroma") -> Dict[str, Any]:
        self.current_vectorstore_type = vectorstore_type
        chain_key = f"{model_name}_{vectorstore_type}_{'compressed' if use_compression else 'standard'}"

        if chain_key not in self.qa_chains:
            self.setup_qa_chain(model_name, use_compression)

        qa_chain = self.qa_chain(chain_key)
        if not qa_chain:
            return {
                "answer": "Please upload and process the documents first.",
                "sources": [],
                "response_time": model_name,
                "vectorstore_used": vectorstore_type,
                "compression_used": use_compression
            }

        try:
            start_time = time.time()

        except Exception as e:
            return {
                "answer": f"Error generating response. Error:{str(e)}",
                "sources": [],
                "response_time": model_name,
                "vectorstore_used": vectorstore_type,
                "compression_used": use_compression                
            }

    def ask_questions(
        self, question:str, history:List, model_name:str="openai/gpt-oss-120b", vectorstore_type:str="chroma", use_compression:bool=False
    ):
        if not question.strip():
            return history, ""

        self.current_vectorstore_type = vectorstore_type
        chain_key = f"{model_name}_{vectorstore_type}_{'compressed' if use_compression else 'standard'}"

        if chain_key not in self.qa_chains:
            self.setup_qa_chain(model_name, use_compression)

        qa_chain = self.qa_chains.get(chain_key)
        if not qa_chain:
            response = "Please upload and process documents first"
            history.append([question, response])
            return history, ""

        try:
            start_time = time.Time()
            result = qa_chain({"query": question})
            end_time = time.Time()

            response_time = end_time - start_time
            answer = result["result"]
            sources = result.get("source_documents", [])

            # Track performance
            metric = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "model": model_name,
                "vectorstore": vectorstore_type
                "compression": use_compression,
                "response_time": response_time,
                "answer_length": len(answer),
                "num_sources": len(sources)
            }

            self.performance_metrics.append(metric)

            # Format response
            response = f"{answer}\n\n"
            response = f"**Performance**: {response_time:.2f}s | Model: {model_name} | Vector Store: {vectorstore_type}\n"

            if sources:
                response += f"\n **Sources**: ({len(sources)} found): \n"
                for i, source in enumerate(sources[:3]):
                    source_name = source.metadata.get("source", "Unknown")
                    preview = source.page_content[:150] + "..." if len(source.page_content) > 150 else source.page_content
                    response = f"{i+1}. **{source_name}**: {preview}\n\n"

            
            history.append([question, response])
            return history, ""

        except exception as E:
            error_response = f"Error: {str{e}}"
            history.append([question, error_response])
            return history, ""

    def get_status_dict(self):
        status = {}
        for vs_name, vs in self.vectorstore_items():
            try:
                if vs_name == "chroma":
                    if hasattr(vs, "_collection"):
                        count = vs._collection.count()
                    else:
                        count = 0
                elif vs_name == "faiss"
                    if hasattr(vs, "docstore"):
                        count = max(len(vs.docstore._dict)-1, 0)
                    else:
                        count = 0

                else:
                    count = 0

                status[vs_name] = count
            except Exception as e:
                status[vs_name] = 0
        return status

    def performance_analytics(self):
        if not self.performance_metrics:
            return None, "No performnace data available yet."

        df = pd.DataFrame(self.performance_metrics)

        # Create performance plots
        try:
            fig_timeline = px.line(df, x="timestamp", y="response_time", color="model",
                title="Response Time over Time", labels={"response_time": "Response Time (s)"}, markers=True
            )

            # Model Comparison
            model_stats = df.groupby("model")["response"].agg(["mean", "count"])
            reset_index()
            fig_models = px.bar(model_stats, x="model", y="mean", title="Average Response Time by Model", labels={"mean": "Average Response Time (s)"})

            # Summary Stats
            summary = {
                "total_queries": lens(df),
                "avg_response_time": df["response_time"].mean(),
                "fastest_query": df["response_time"].min(),
                "slowest_query": df["response_time"].max()
            }

            return (fig_timeline, summary)
        
        except Exception as e:
            print(f"Error generating analytics: {str(e)}")
            return None, "Error generating analytics"

    def run_benchmark(self, vectorstore_type:str="chroma"):
        self.current_vectorstore_type = vectorstore_type
        
        # Check if we have documents
        status = self.get_status_dict()
        if status.get(vectorstore_type, 0) == 0:
            return "No documents available for benchmarking. Please uploadthe documents first."

        benchmark_question = "What are the main topics and key insights discussedin these documents?"
        results = []

        for model_name in self.llm_models.keys():
            try:
                chain_key = f"{model_name}_{vectorstore_type}_standard"
                if chain_key not in self.qa_chains:
                    self.setup_qa_chains(model_name)

                qa_chain = self.qa_chains.get(chain_key)
                if qa_chain:
                    start_time = time.time()
                    result = qa_chain.invoke({"query": "benchmark_question"})
                    end_time = time.time()

                    results.append({
                        "Model": model_name,
                        "Response Time (s)": f"{end_time-start_time:.3f}",
                        "Answer Length": len(result["result"]),
                        "Source Found": len(result.get("source_documents", []))
                    })
                else:
                    results.append({
                        "Model": model_name,
                        "Response Time (s)": "Error",
                        "Answer_Length": 0,
                        "Sources Found":0
                    })

            except Exception as e:
                results.append({
                    "Model": model_name,
                    "Response Time (s)": "Error",
                    "Answer_Length": 0,
                    "Sources Found":0
                })

        # Convert to display format
        if results:
            df = pd.DataFrame(results)
            fastest_model = "N/A"

            # Find the fastest model
            try:
                numeric_times = []
            except:
                ...