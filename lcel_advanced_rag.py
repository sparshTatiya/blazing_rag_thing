"""
Advanced Multi-Model LCEL-Based RAG System
Modern LangChain Expression Language
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

# LangChain imports - LCEL style
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)

# Environment and config
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class AdvancedLCELRAG:
    def __init__(self):
        self.vectorstores = {}
        self.rag_chains = {}
        self.embeddings = None
        self.llm_models = {}
        self.current_model = "OPENAI"
        self.current_vectorstore = "chroma"
        self.performance_metrics = []
        self.setup_components()
    
    def setup_components(self):
        """Initialize advanced components with LCEL support"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize multiple Groq models
            self.llm_models = {
                "OPENAI": ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model = "openai/gpt-oss-120b",
                    temperature=0.1,
                    max_tokens=1024
                ),
                "LLAMA": ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_tokens=1024
                ),
                "QWEN": ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model = "qwen/qwen3-32b",
                    temperature=0.1,
                    max_tokens=1024
                ),
                "MOONSHOT": ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model = "moonshotai/kimi-k2-instruct-0905",
                    temperature=0.1,
                    max_tokens=1024
                )
            }
            
            # Initialize vector stores
            self.setup_vectorstores()
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return False
        return True
    
    def setup_vectorstores(self):
        """Setup multiple vector stores"""
        # ChromaDB
        self.vectorstores["chroma"] = Chroma(
            persist_directory="./chroma_db_lcel_advanced",
            embedding_function=self.embeddings
        )
        
        # FAISS
        try:
            if os.path.exists("./faiss_index_lcel"):
                self.vectorstores["faiss"] = FAISS.load_local(
                    "./faiss_index_lcel",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                sample_text = ["Initialization"]
                sample_embeddings = self.embeddings.embed_documents(sample_text)
                self.vectorstores["faiss"] = FAISS.from_embeddings(
                    [(sample_text[0], sample_embeddings[0])],
                    self.embeddings
                )
        except Exception as e:
            st.warning(f"FAISS initialization: {e}")
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format documents for context with metadata"""
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            formatted.append(f"[Source: {source}]\n{content}")
        return "\n\n---\n\n".join(formatted)
    
    def create_custom_prompt(self, style: str = "standard") -> ChatPromptTemplate:
        """Create different prompt styles"""
        prompts = {
            "standard": """You are an expert assistant analyzing documents. Use the following context to answer the question accurately.

Context:
{context}

Question: {question}

Answer:""",
            
            "detailed": """You are a thorough research assistant. Analyze the provided context deeply and provide a comprehensive answer.

Context:
{context}

Question: {question}

Provide a detailed answer with:
1. Direct answer to the question
2. Supporting evidence from the context
3. Any relevant nuances or limitations

Answer:""",
            
            "concise": """You are a concise assistant. Provide brief, direct answers based on the context.

Context:
{context}

Question: {question}

Brief Answer:"""
        }
        
        return ChatPromptTemplate.from_template(prompts.get(style, prompts["standard"]))
    
    def setup_lcel_chain(
        self, 
        model_name: str = "llama3-8b",
        vectorstore_type: str = "chroma",
        prompt_style: str = "standard",
        use_reranking: bool = False
    ):
        """Setup advanced LCEL chain with multiple options"""
        
        vectorstore = self.vectorstores.get(vectorstore_type)
        if not vectorstore:
            return
        
        # Check if vectorstore has documents
        try:
            if vectorstore_type == "chroma":
                has_docs = hasattr(vectorstore, '_collection') and vectorstore._collection.count() > 0
            elif vectorstore_type == "faiss":
                has_docs = hasattr(vectorstore, 'docstore') and len(vectorstore.docstore._dict) > 1
            else:
                has_docs = False
        except Exception:
            has_docs = False
        
        if not has_docs:
            return
        
        # Create retriever with MMR for diversity
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )
        
        # Get LLM and prompt
        llm = self.llm_models[model_name]
        prompt = self.create_custom_prompt(prompt_style)
        
        # Build LCEL chain with advanced features
        # Using RunnableParallel for parallel execution
        chain_key = f"{model_name}_{vectorstore_type}_{prompt_style}"
        
        # Basic LCEL chain
        basic_chain = (
            RunnableParallel(
                {
                    "context": retriever | RunnableLambda(self.format_docs),
                    "question": RunnablePassthrough()
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Chain with sources
        chain_with_sources = RunnableParallel(
            {
                "answer": basic_chain,
                "source_documents": retriever,
                "question": RunnablePassthrough()
            }
        )
        
        self.rag_chains[chain_key] = {
            "basic": basic_chain,
            "with_sources": chain_with_sources
        }
    
    def process_documents(self, uploaded_files, chunk_strategy: str = "standard") -> Dict[str, Any]:
        """Process documents with different chunking strategies"""
        if not uploaded_files:
            return {"success": False, "message": "No files uploaded"}
        
        try:
            chunk_configs = {
                "standard": {"chunk_size": 1000, "overlap": 200},
                "small": {"chunk_size": 500, "overlap": 100},
                "large": {"chunk_size": 1500, "overlap": 300}
            }
            
            config = chunk_configs.get(chunk_strategy, chunk_configs["standard"])
            all_documents = []
            file_info = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    if uploaded_file.name.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file_path)
                    elif uploaded_file.name.endswith('.txt'):
                        loader = TextLoader(tmp_file_path, encoding='utf-8')
                    else:
                        continue
                    
                    documents = loader.load()
                    
                    for doc in documents:
                        doc.metadata.update({
                            'source': uploaded_file.name,
                            'upload_time': datetime.now().isoformat(),
                            'chunk_strategy': chunk_strategy
                        })
                    
                    all_documents.extend(documents)
                    file_info.append(f"✅ {uploaded_file.name}: {len(documents)} pages")
                    
                finally:
                    os.unlink(tmp_file_path)
            
            if not all_documents:
                return {"success": False, "message": "No valid documents"}
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["overlap"],
                length_function=len
            )
            
            split_documents = text_splitter.split_documents(all_documents)
            
            # Add to vector stores
            for vs_name, vs in self.vectorstores.items():
                try:
                    if vs_name == "chroma":
                        vs.add_documents(split_documents)
                        vs.persist()
                    elif vs_name == "faiss":
                        vs.add_documents(split_documents)
                        vs.save_local("./faiss_index_lcel")
                except Exception as e:
                    st.warning(f"Error adding to {vs_name}: {e}")
            
            # Setup chains for all configurations
            for model_name in self.llm_models.keys():
                for vs_type in ["chroma", "faiss"]:
                    for prompt_style in ["standard", "detailed", "concise"]:
                        self.setup_lcel_chain(model_name, vs_type, prompt_style)
            
            return {
                "success": True,
                "message": f"Processed {len(uploaded_files)} files, created {len(split_documents)} chunks",
                "file_info": file_info,
                "chunk_count": len(split_documents)
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def ask_question(
        self,
        question: str,
        model_name: str = "llama3-8b",
        vectorstore_type: str = "chroma",
        prompt_style: str = "standard",
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """Ask question using LCEL chain with full configuration"""
        
        chain_key = f"{model_name}_{vectorstore_type}_{prompt_style}"
        
        if chain_key not in self.rag_chains:
            self.setup_lcel_chain(model_name, vectorstore_type, prompt_style)
        
        chains = self.rag_chains.get(chain_key)
        if not chains:
            return {
                "answer": "Please upload and process documents first.",
                "sources": [],
                "response_time": 0
            }
        
        try:
            start_time = time.time()
            
            if return_sources:
                result = chains["with_sources"].invoke(question)
                answer = result['answer']
                sources = result.get('source_documents', [])
            else:
                answer = chains["basic"].invoke(question)
                sources = []
            
            response_time = time.time() - start_time
            
            # Track metrics
            self.performance_metrics.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "model": model_name,
                "vectorstore": vectorstore_type,
                "prompt_style": prompt_style,
                "response_time": response_time
            })
            
            return {
                "answer": answer,
                "sources": sources,
                "response_time": response_time,
                "model": model_name,
                "vectorstore": vectorstore_type,
                "prompt_style": prompt_style
            }
            
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "response_time": 0
            }
    
    def stream_question(self, question: str, model_name: str = "llama3-8b", 
                       vectorstore_type: str = "chroma", prompt_style: str = "standard"):
        """Stream response using LCEL"""
        chain_key = f"{model_name}_{vectorstore_type}_{prompt_style}"
        
        if chain_key not in self.rag_chains:
            self.setup_lcel_chain(model_name, vectorstore_type, prompt_style)
        
        chains = self.rag_chains.get(chain_key)
        if not chains:
            yield "Please upload documents first."
            return
        
        try:
            for chunk in chains["basic"].stream(question):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {}
        for vs_name, vs in self.vectorstores.items():
            try:
                if vs_name == "chroma":
                    count = vs._collection.count() if hasattr(vs, '_collection') else 0
                elif vs_name == "faiss":
                    count = max(0, len(vs.docstore._dict) - 1) if hasattr(vs, 'docstore') else 0
                else:
                    count = 0
                stats[vs_name] = count
            except:
                stats[vs_name] = 0
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analytics"""
        if not self.performance_metrics:
            return {}
        
        df = pd.DataFrame(self.performance_metrics)
        return {
            "total_queries": len(df),
            "avg_response_time": df['response_time'].mean(),
            "fastest": df['response_time'].min(),
            "slowest": df['response_time'].max(),
            "by_model": df.groupby('model')['response_time'].mean().to_dict()
        }
    
    def clear_all(self):
        """Clear all databases"""
        try:
            for vs_name, vs in self.vectorstores.items():
                if vs_name == "chroma" and hasattr(vs, 'delete_collection'):
                    vs.delete_collection()
                elif vs_name == "faiss" and hasattr(vs, 'docstore'):
                    vs.docstore._dict.clear()
            
            self.setup_vectorstores()
            self.rag_chains.clear()
            self.performance_metrics = []
            
            return "✅ All databases cleared"
        except Exception as e:
            return f"❌ Error: {str(e)}"

def main():
    st.set_page_config(
        page_title="🚀 Advanced LCEL RAG",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("🚀 Advanced Multi-Model LCEL RAG System")
    st.markdown("*Modern LangChain Expression Language with advanced features*")
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("🔑 Please set your GROQ_API_KEY")
        st.stop()
    
    # Initialize
    if 'advanced_rag' not in st.session_state:
        with st.spinner("🚀 Initializing advanced LCEL system..."):
            st.session_state.advanced_rag = AdvancedLCELRAG()
    
    rag = st.session_state.advanced_rag
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ LCEL Configuration")
        
        # Model selection
        model = st.selectbox(
            "🤖 Model",
            options=list(rag.llm_models.keys()),
            index=0
        )
        
        # Vector store
        vectorstore = st.selectbox(
            "🗂️ Vector Store",
            options=["chroma", "faiss"],
            index=0
        )
        
        # Prompt style
        prompt_style = st.selectbox(
            "📝 Prompt Style",
            options=["standard", "detailed", "concise"],
            index=0
        )
        
        # Streaming
        use_streaming = st.checkbox("🎬 Enable Streaming", value=False)
        
        st.divider()
        
        # Stats
        st.header("📊 Statistics")
        stats = rag.get_stats()
        for vs_name, count in stats.items():
            st.metric(f"{vs_name.title()}", count)
        
        st.divider()
        
        # File upload
        st.header("📤 Upload")
        uploaded_files = st.file_uploader(
            "Documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True
        )
        
        chunk_strategy = st.selectbox(
            "Chunking",
            options=["standard", "small", "large"]
        )
        
        if uploaded_files:
            if st.button("🔄 Process", type="primary"):
                with st.spinner("Processing..."):
                    result = rag.process_documents(uploaded_files, chunk_strategy)
                    if result["success"]:
                        st.success(result["message"])
                        st.rerun()
                    else:
                        st.error(result["message"])
        
        # Clear
        if any(stats.values()):
            if st.button("🗑️ Clear All"):
                msg = rag.clear_all()
                st.info(msg)
                st.rerun()
    
    # Main interface
    st.header("💬 Intelligent Q&A")
    
    if not any(rag.get_stats().values()):
        st.info("📁 Upload documents to start")
    else:
        # Show LCEL chain info
        with st.expander("🔗 Current LCEL Chain Configuration"):
            st.code(f"""
# Current Chain:
chain = (
    RunnableParallel({{
        "context": {vectorstore}_retriever | format_docs,
        "question": RunnablePassthrough()
    }})
    | {prompt_style}_prompt
    | {model}_llm
    | StrOutputParser()
)

# Features:
✅ Composable with pipe operator
✅ Parallel context retrieval
✅ {prompt_style.title()} prompt style
✅ {"Streaming enabled" if use_streaming else "Standard mode"}
            """, language="python")
        
        # Question input
        question = st.text_input(
            "Your Question:",
            placeholder="Ask anything about your documents...",
            key="question_input"
        )
        
        # Quick questions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Summarize"):
                st.session_state.question_input = "Provide a comprehensive summary of the key points."
                st.rerun()
        with col2:
            if st.button("🔍 Methods"):
                st.session_state.question_input = "What methodologies are discussed?"
                st.rerun()
        with col3:
            if st.button("💡 Insights"):
                st.session_state.question_input = "What are the main insights and conclusions?"
                st.rerun()
        
        # Process question
        if question:
            st.subheader("🎯 Answer")
            
            if use_streaming:
                # Streaming with LCEL
                answer_placeholder = st.empty()
                full_response = ""
                
                start_time = time.time()
                for chunk in rag.stream_question(question, model, vectorstore, prompt_style):
                    full_response += chunk
                    answer_placeholder.markdown(full_response + "▌")
                answer_placeholder.markdown(full_response)
                
                response_time = time.time() - start_time
                st.success(f"⚡ Streamed in {response_time:.2f}s")
                
            else:
                # Standard with sources
                with st.spinner("🚀 Generating with LCEL..."):
                    result = rag.ask_question(
                        question, model, vectorstore, prompt_style
                    )
                
                st.write(result['answer'])
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⚡ Time", f"{result['response_time']:.2f}s")
                with col2:
                    st.metric("🤖 Model", result['model'])
                with col3:
                    st.metric("🗂️ Store", result['vectorstore'])
                with col4:
                    st.metric("📝 Style", result['prompt_style'])
                
                # Sources
                if result['sources']:
                    with st.expander(f"📚 Sources ({len(result['sources'])})"):
                        for i, source in enumerate(result['sources'][:3]):
                            st.markdown(f"**{i+1}. {source.metadata.get('source', 'Unknown')}**")
                            st.text(source.page_content[:250] + "...")
                            st.divider()
        
        # Performance summary
        perf = rag.get_performance_summary()
        if perf:
            with st.expander("📊 Performance Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Queries", perf['total_queries'])
                with col2:
                    st.metric("Avg Time", f"{perf['avg_response_time']:.3f}s")
                with col3:
                    st.metric("Fastest", f"{perf['fastest']:.3f}s")

if __name__ == "__main__":
    main()
