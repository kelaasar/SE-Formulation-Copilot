"""
Chat JPL - Main Streamlit Application
Organized and modular codebase for JPL's AI assistant interface.
"""

import streamlit as st
import time
import os
import json  # Add missing import
import requests
import numpy as np
import pickle
import hashlib
import asyncio
from datetime import datetime, timedelta

# Import the centralized model handler
try:
    from utils.model_handler import get_chat_client, get_embeddings_client, get_workflow_kb_and_client
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LLM_AVAILABLE = True
except ImportError as e:
    st.error(f"LLM dependencies not available: {e}")
    LLM_AVAILABLE = False

# Import our organized modules
from utils.styles import apply_custom_css
from utils.data_persistence import load_conversations, save_conversations, load_user_profile, load_custom_knowledge_bases, load_rag_settings, load_user_preferences, save_user_preferences
from utils.vector_db import load_vector_db, create_chat_id
from utils.reference_handler import render_sources_ui, show_source_modal, show_all_sources_modal
from components.sidebar import render_sidebar
from components.top_nav import render_top_nav
from components.profile import render_profile_page
from components.settings import render_settings_page

# Try to import UI modules, handle gracefully if missing
try:
    from workflows.proposal_assistant import proposal_assistant_ui
    PROPOSAL_ASSISTANT_AVAILABLE = True
except ImportError:
    PROPOSAL_ASSISTANT_AVAILABLE = False

# Configure the page
st.set_page_config(
    page_title="SE Copilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS styling
apply_custom_css()

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    
    # Check URL query parameters for page state
    query_params = st.query_params
    page_from_url = query_params.get("page", "chat")
    
    # Load user preferences first
    user_preferences = load_user_preferences()
    
    # Initialize all required session state variables with defaults
    defaults = {
        "messages": [],
        "conversations": [],
        "current_conversation": None,
        "current_chat_id": None,
        "show_settings": False,
        "current_page": page_from_url,
        "knowledge_bases": [],  # Only one knowledge base system
        "new_kb_name": "",
        "new_kb_description": "",
        "chunking_url": "http://localhost:9876",
        "user_profile": {},
        "rag_settings": {},
        "user_preferences": user_preferences
    }
    
    # Set defaults for any missing keys
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Load data that might take time or fail
    try:
        if not st.session_state.conversations:
            st.session_state.conversations = load_conversations()
    except Exception as e:
        print(f"Error loading conversations: {e}")
        st.session_state.conversations = []
    
    # Load knowledge bases from unified system
    try:
        kb_file_path = "databases/user_preferences.json"
        should_reload = False
        
        # Check if we should reload (file doesn't exist in session state or file is newer)
        if not st.session_state.knowledge_bases:
            should_reload = True
        elif os.path.exists(kb_file_path):
            # Check if file has been modified since last load
            file_mtime = os.path.getmtime(kb_file_path)
            last_loaded = st.session_state.get("kb_last_loaded", 0)
            if file_mtime > last_loaded:
                should_reload = True
        
        if should_reload:
            all_kbs = []
            if os.path.exists(kb_file_path):
                with open(kb_file_path, "r") as f:
                    user_prefs = json.load(f)
                    # Extract just the custom_knowledge_bases section
                    all_kbs = user_prefs.get("custom_knowledge_bases", [])
                st.session_state.kb_last_loaded = os.path.getmtime(kb_file_path)
            
            st.session_state.knowledge_bases = all_kbs
    except Exception as e:
        print(f"Error loading knowledge bases: {e}")
        st.session_state.knowledge_bases = []
    
    # Load other data
    try:
        if not st.session_state.user_profile:
            st.session_state.user_profile = load_user_profile()
    except Exception as e:
        print(f"Error loading user profile: {e}")
        st.session_state.user_profile = {"name": "User", "role": "Engineer", "location": "JPL"}
    
    try:
        if not st.session_state.rag_settings:
            st.session_state.rag_settings = load_rag_settings()
    except Exception as e:
        print(f"Error loading RAG settings: {e}")
        st.session_state.rag_settings = {
            "full_context_mode": False,
            "hybrid_search": True,
            "reranking_engine": False,
            "reranking_model": "BAAI/bge-reranker-v2-m3",
            "top_k": 5,
            "top_k_reranker": 3,
            "relevance_threshold": 0.5,
            "bm25_weight": 0.5
        }
    
    # Restore last conversation if user was in a conversation when they refreshed
    if not st.session_state.current_conversation and not st.session_state.messages:
        last_conversation_title = user_preferences.get("last_conversation", None)
        if last_conversation_title and st.session_state.conversations:
            # Find the conversation in the loaded conversations
            for conv in st.session_state.conversations:
                if conv.get('title') == last_conversation_title:
                    # Restore this conversation
                    st.session_state.current_conversation = conv['title']
                    st.session_state.current_chat_id = conv.get('chat_id')
                    st.session_state.messages = conv.get('messages', [])
                    print(f"[INIT] Restored conversation: {last_conversation_title}")
                    break
    
    # Backward compatibility
    if "user_name" in st.session_state and isinstance(st.session_state.user_profile, dict):
        if st.session_state.user_profile.get('name') != st.session_state.get('user_name'):
            st.session_state.user_profile['name'] = st.session_state.get('user_name', 'User')

# Initialize LLM clients
@st.cache_resource
def get_llm_client():
    if LLM_AVAILABLE:
        # Use main_chat workflow configuration for cached client
        return get_chat_client("main_chat")
    else:
        return None

@st.cache_resource
def get_embedding_client():
    if LLM_AVAILABLE:
        return get_embeddings_client()
    else:
        return None

# Main application logic
def main():
    """Main application entry point"""
    
    # Initialize session state FIRST before any other operations
    initialize_session_state()
    
    # Show top navigation on every page
    render_top_nav()
    
    # Render sidebar
    render_sidebar()
    
    # Page routing
    route_pages()

def route_pages():
    """Handle page routing based on current_page state"""
    
    if st.session_state.current_page == "proposal_assistant":
        # Add margin for top nav
        st.markdown('<div style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
        render_proposal_assistant()
        
    elif st.session_state.current_page == "profile":
        # Add margin for top nav
        st.markdown('<div style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
        render_profile_page()
        
    elif st.session_state.current_page == "settings":
        # Add margin for top nav
        st.markdown('<div style="margin-top: 4rem;"></div>', unsafe_allow_html=True)
        render_settings_page()
        
    else:  # Default to chat
        render_chat_interface()

def render_proposal_assistant():
    """Render the Proposal Assistant page"""
    if PROPOSAL_ASSISTANT_AVAILABLE:
        proposal_assistant_ui()
    else:
        st.error("Proposal Assistant module not available")

def render_chat_interface():
    """Render the main chat interface"""
    
    # Check if we need to show an individual source modal from the "All Sources" dialog
    for key in list(st.session_state.keys()):
        if key.startswith("show_source_") and isinstance(st.session_state[key], dict):
            source_info = st.session_state[key]
            del st.session_state[key]  # Remove it so it doesn't trigger again
            show_source_modal(
                source_info["filename"], 
                source_info["chunks"], 
                source_info["file_index"],
                source_info.get("from_all_sources", False),
                source_info.get("files_dict"),
                source_info.get("message_index")
            )
            break
        elif key.startswith("show_all_sources_") and isinstance(st.session_state[key], dict):
            # Handle back button from individual source to "All Sources" dialog
            files_dict = st.session_state[key]
            message_index = key.split("_")[-1]
            del st.session_state[key]  # Remove it so it doesn't trigger again
            show_all_sources_modal(files_dict, int(message_index))
            break
    
    # Welcome content - hide immediately when generating response or when messages exist
    if not st.session_state.messages and not st.session_state.get("generating_response", False):
        render_welcome_screen()
    else:
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

        # Chat messages
        render_chat_messages()
        
        # Chat input
        render_chat_input()

    # Add bottom margin after chat input
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

def render_welcome_screen():
    """Render the welcome screen when no messages exist"""
    print(f"[Main chat] Rendered UI")
    st.markdown("""
    <div class="welcome-content">
        <h1 class="main-header" style="margin-top: 5rem; font-size: 50px;">Welcome to Systems Engineering Copilot</h1>
        <p class="subtitle" style="margin-bottom: 1rem;">
            Your personal AI assistant for system engineering.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add chat input section at the top
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    render_chat_input()
    
    # Add some spacing before the workflows
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

def render_chat_messages():
    """Render all chat messages using Streamlit's native chat components"""
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                # Only handle text messages now (no image support)
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Show sources if available with modern UI
                sources = message.get("sources")
                if sources:
                    render_sources_ui(sources, i)
                
                # If this is the latest assistant message and temp_sources exist, attach them
                if i == len(st.session_state.messages) - 1 and st.session_state.get("temp_sources"):
                    message["sources"] = st.session_state.temp_sources
                    del st.session_state["temp_sources"]

def render_chat_input():
    """Render the chat input form with file upload capability"""
    
    # Check if we need to generate a response
    if st.session_state.get("generating_response", False):
        st.session_state.generating_response = False
        
        # Generate AI response
        with st.spinner("Thinking..."):
            try:
                # Get the last user message
                last_user_message = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "user":
                        last_user_message = msg["content"]
                        break
                
                if last_user_message:
                    response = generate_ai_response(last_user_message)
                    
                    # Get sources if available
                    sources = st.session_state.get("temp_sources", [])
                    if "temp_sources" in st.session_state:
                        del st.session_state.temp_sources
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources
                    })
                    
                    # Save conversation
                    save_current_conversation()
                    
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error processing your request. Please try again.",
                    "sources": []
                })
        
        # Add margin below the thinking spinner
        st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
        st.rerun()
    
    # File upload section (removed file dropdown display for cleaner UI)
    
    # Check if last assistant message has no sources to add top margin
    add_top_margin = False
    if st.session_state.messages:
        # Get the last assistant message
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                if not sources:  # No sources available
                    add_top_margin = True
                break
    
    # Add top margin if last assistant message has no sources
    if add_top_margin:
        st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)
    
    # File uploader - single unified uploader for all file types
    uploaded_files = None
    try:
        # Make file uploader key chat-specific to isolate uploads per conversation
        chat_id = st.session_state.get('current_chat_id', 'default')
        file_upload_key = f"chat_file_upload_{chat_id}" if chat_id else "chat_file_upload_new"
        
        uploaded_files = st.file_uploader(
            "Upload files",
            type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'md', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key=file_upload_key,
            label_visibility="collapsed",
            help="📄 **Documents**: PDF, Word, Excel, PowerPoint, text files for context and analysis\n📷 **Images**: JPG, PNG for visual analysis with AI"
        )
    except Exception as e:
        st.warning("⚠️ File upload temporarily unavailable. You can still chat normally.")
        st.caption(f"Technical details: {str(e)}")
        uploaded_files = None

    # Process uploaded files - separate handling for documents and images
    if uploaded_files:
        # Separate documents from images
        documents = []
        images = []
        
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type.lower() if uploaded_file.type else ""
            file_name = uploaded_file.name.lower()
            
            # Check if it's an image
            if (file_type.startswith('image/') or 
                any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])):
                images.append(uploaded_file)
            else:
                documents.append(uploaded_file)
        
        # Process documents for RAG
        if documents:
            process_uploaded_files(documents)
            
        # Store images in session state for vision processing
        if images:
            from utils.image_processing import process_uploaded_image, validate_image_file
            processed_images = []
            
            for image_file in images:
                if validate_image_file(image_file):
                    processed_image = process_uploaded_image(image_file)
                    if processed_image:
                        processed_images.append(processed_image)
                else:
                    st.warning(f"⚠️ Unsupported image format: {image_file.name}")
            
            if processed_images:
                st.session_state.uploaded_images = processed_images
            else:
                st.session_state.uploaded_images = []
        else:
            st.session_state.uploaded_images = []
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0

    # Chat input form - use clear_on_submit=True
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_submit = st.columns([5, 1])
        
        with col_input:
            user_input = st.text_area(
                "Message",
                placeholder="Ask me anything...",
                label_visibility="collapsed",
                key=f"chat_input_field_{st.session_state.chat_input_key}"
            )
        
        with col_submit:
            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Send", width='stretch')
    
    if submitted and user_input:
        # Increment the key to reset the input field on next render
        st.session_state.chat_input_key += 1
        handle_user_message(user_input, uploaded_files)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to chat embeddings with progress tracking"""
    if not uploaded_files:
        return
    
def process_uploaded_files(uploaded_files):
    """Process uploaded files and add them to chat embeddings with progress tracking"""
    if not uploaded_files:
        return
    
    # Process each file without showing progress under the upload widget
    for uploaded_file in uploaded_files:
        # Make file processing key chat-specific to isolate files per conversation
        chat_id = st.session_state.get('current_chat_id', 'default')
        file_key = f"processed_{chat_id}_{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
        if file_key not in st.session_state:
            file_size = getattr(uploaded_file, 'size', 0)
            size_mb = file_size / (1024*1024) if file_size > 0 else 0
            
            # Show different spinner message for large files
            if size_mb > 100:
                spinner_msg = f"Processing large file {uploaded_file.name} ({size_mb:.1f} MB) - This may take several minutes..."
            elif size_mb > 50:
                spinner_msg = f"Processing {uploaded_file.name} ({size_mb:.1f} MB) - This may take a few minutes..."
            else:
                spinner_msg = f"Processing {uploaded_file.name}..."
            
            with st.spinner(spinner_msg):
                try:
                    # Send file to chunking endpoint for processing
                    chunks = process_uploaded_file_content(uploaded_file)
                    if chunks:
                        # Create chat ID if not exists
                        if st.session_state.current_chat_id is None:
                            if st.session_state.current_conversation:
                                st.session_state.current_chat_id = create_chat_id(st.session_state.current_conversation)
                                print(f"[FILE_UPLOAD] Created chat_id from existing conversation: {st.session_state.current_chat_id}")
                            else:
                                # Create temporary chat ID for new conversation
                                temp_title = f"temp_chat_{int(time.time())}"
                                st.session_state.current_chat_id = create_chat_id(temp_title)
                                print(f"[FILE_UPLOAD] Created temporary chat_id: {st.session_state.current_chat_id}")
                        else:
                            print(f"[FILE_UPLOAD] Using existing chat_id: {st.session_state.current_chat_id}")
                        
                        # Add to chat embeddings
                        print(f"[FILE_UPLOAD] Adding {uploaded_file.name} to chat_id: {st.session_state.current_chat_id}")
                        success, num_chunks = add_documents_to_chat_embeddings(
                            st.session_state.current_chat_id, 
                            chunks, 
                            uploaded_file.name
                        )
                        
                        if success:
                            st.session_state[file_key] = True
                        else:
                            st.error(f"❌ Failed to upload {uploaded_file.name}")
                    else:
                        st.error(f"❌ Could not process {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

def process_uploaded_file_content(uploaded_file):
    """Process uploaded file using the chunking endpoint with support for large files"""
    try:
        print(f"[MAIN] Processing file: {uploaded_file.name} (Size: {getattr(uploaded_file, 'size', 'unknown')})")
        
        file_size = getattr(uploaded_file, 'size', 0)
        if file_size > 0:
            print(f"[MAIN] File size: {file_size / (1024*1024):.2f} MB")
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        chunking_url = st.session_state.get("chunking_url", "http://localhost:9876")
        print(f"[MAIN] Processing file: {uploaded_file.name} (size: {file_size / (1024*1024):.2f} MB)")
        print(f"[MAIN] Sending request to chunking service at {chunking_url}/process")
        
        response = requests.post(
            f"{chunking_url}/process",
            files=files,
            stream=True  # Enable streaming for large files
        )
        
        print(f"[MAIN] Received response with status code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"[MAIN] Parsing JSON response...")
            result = response.json()
            print(f"[MAIN] Response type: {type(result)}")
            print(f"[MAIN] Response sample: {result[:2] if isinstance(result, list) else result}")
            
            # Handle different response formats
            if isinstance(result, list):
                # Check if it's a list of objects with page_content
                if result and isinstance(result[0], dict) and "page_content" in result[0]:
                    chunks = [item["page_content"] for item in result]
                    print(f"[MAIN] Extracted {len(chunks)} chunks from page_content format")
                else:
                    # If response is directly a list of chunks
                    chunks = result
                    print(f"[MAIN] Received direct list of {len(chunks)} chunks")
            elif isinstance(result, dict):
                # If response is a dictionary with chunks key
                chunks = result.get("chunks", [])
                print(f"[MAIN] Received dict with {len(chunks)} chunks")
            else:
                print(f"[MAIN] Unexpected response format: {type(result)}")
                chunks = []
                
            print(f"[MAIN] Successfully processed {len(chunks)} chunks from endpoint")
            
            # Debug: Show sample of first few chunks
            if chunks:
                print(f"[MAIN] Sample chunks:")
                for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                    chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    print(f"[MAIN]   Chunk {i+1}: {chunk_preview}")
            
            return chunks
        else:
            error_msg = f"Chunking service error: {response.status_code}"
            print(f"[MAIN] {error_msg}")
            print(f"[MAIN] Server response: {response.text[:500]}")
            st.error(error_msg)
            return []
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        print(f"[MAIN] Error processing {uploaded_file.name}: {e}")
        return []

def add_documents_to_chat_embeddings(chat_id, chunks, filename):
    """Add document chunks to chat-specific embeddings"""
    try:
        from utils.vector_db import load_chat_embeddings, save_chat_embeddings
        import numpy as np
        import faiss
        
        if not LLM_AVAILABLE:
            return False, 0
        
        # Get embedding client
        embedding_client = get_embedding_client()
        print(f"[EMBEDDINGS] Starting embedding generation for {len(chunks)} chunks")
        
        # Generate embeddings for chunks in batches
        batch_size = 8
        embeddings = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx:batch_idx+batch_size]
            batch_num = (batch_idx // batch_size) + 1
            print(f"[EMBEDDINGS] Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                batch_embeddings = embedding_client.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                print(f"[EMBEDDINGS] Successfully generated embeddings for batch {batch_num}")
            except Exception as e:
                print(f"[EMBEDDINGS] Error generating embeddings for batch {batch_num}: {e}")
                st.error(f"Error generating embeddings: {e}")
                return False, 0
        
        print(f"[EMBEDDINGS] Completed embedding generation for all {len(embeddings)} chunks")
        print(f"[FAISS] Converting embeddings to numpy array...")
        
        embeddings_array = np.array(embeddings).astype('float32')
        print(f"[FAISS] Embeddings array shape: {embeddings_array.shape}")
        
        # Load existing chat embeddings or create new
        print(f"[FAISS] Loading existing chat embeddings for chat_id: {chat_id}")
        index, metadata = load_chat_embeddings(chat_id)
        
        if index is None:
            # Create new FAISS index
            dimension = embeddings_array.shape[1]
            print(f"[FAISS] Creating new FAISS index with dimension: {dimension}")
            index = faiss.IndexFlatIP(dimension)
            metadata = {"chunks": [], "filenames": [], "chunk_ids": [], "source_types": []}
        else:
            # For existing index, check compatibility
            existing_dimension = index.d
            print(f"[FAISS] Existing index found with dimension: {existing_dimension}, new dimension: {embeddings_array.shape[1]}")
            if existing_dimension != embeddings_array.shape[1]:
                st.error(f"Embedding dimension mismatch: index={existing_dimension}, chunk={embeddings_array.shape[1]}")
                return False, 0
        
        # Normalize embeddings for cosine similarity
        print(f"[FAISS] Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        start_id = index.ntotal
        print(f"[FAISS] Adding {len(embeddings_array)} embeddings to index (starting from ID {start_id})")
        index.add(embeddings_array)
        print(f"[FAISS] Index now contains {index.ntotal} total embeddings")
        
        # Update metadata
        print(f"[FAISS] Updating metadata for {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            metadata["chunks"].append(chunk)
            metadata["filenames"].append(filename)
            metadata["chunk_ids"].append(start_id + i)
            metadata["source_types"].append("uploaded_file")
        
        print(f"[FAISS] Metadata now contains {len(metadata['chunks'])} total chunks")
        
        # Save updated embeddings
        print(f"[FAISS] Saving updated embeddings and metadata...")
        save_chat_embeddings(chat_id, index, metadata)
        print(f"[FAISS] Successfully saved embeddings for {filename}")
        
        return True, len(chunks)
        
    except Exception as e:
        st.error(f"Error adding documents to chat embeddings: {e}")
        return False, 0

def handle_user_message(user_input, uploaded_files=None):
    """Process user message and generate response"""
    # No image processing - just use text input directly
    user_message_content = user_input
    
    # Add user message immediately
    st.session_state.messages.append({"role": "user", "content": user_message_content})
    
    # Create conversation if first message
    if not st.session_state.current_conversation:
        conversation_title = user_input[:50] + "..." if len(user_input) > 50 else user_input
        st.session_state.current_conversation = conversation_title
        
        # Only create new chat_id if one doesn't exist (to preserve uploaded files)
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = create_chat_id(conversation_title)
            print(f"[HANDLE_MESSAGE] Created new chat_id: {st.session_state.current_chat_id} for conversation: {conversation_title}")
        else:
            print(f"[HANDLE_MESSAGE] Using existing chat_id: {st.session_state.current_chat_id} for conversation: {conversation_title}")
    
    # Set generating state and rerun to show user message
    st.session_state.generating_response = True
    st.rerun()

def get_current_model_info(workflow="main_chat"):
    """Get information about the currently used model for a specific workflow"""
    try:
        # Use the centralized model handler to get the actual model being used
        selected_kb_name, selected_kb, chat_client = get_workflow_kb_and_client(workflow)
        
        if chat_client and hasattr(chat_client, 'model_name'):
            # Let the client tell us what model it's actually using
            return f"Model: {chat_client.model_name}"
        elif chat_client:
            # If client exists but no model_name attribute, try to detect from type
            client_type = type(chat_client).__name__
            return f"Model: {client_type}"
        else:
            return "Model: Unknown"
            
    except Exception as e:
        print(f"Error getting model info: {e}")
        return "Model: Unknown"

def generate_ai_response(user_input):
    """Generate AI response using configured LLM with RAG from selected knowledge base or uploaded files"""
    if not LLM_AVAILABLE:
        return "I apologize, but the AI chat functionality is currently unavailable due to missing LLM dependencies. Please contact your administrator."
    
    # Import vision processing functions
    from utils.image_processing import is_vision_capable_model, create_vision_message_content
    
    # Extract text from user input and check for uploaded images
    user_text = user_input
    uploaded_images = st.session_state.get("uploaded_images", [])
    
    # Get the appropriate model client for main chat using the centralized model handler
    kb_prefs = st.session_state.user_preferences.get("selected_knowledge_base", {})
    selected_kb_name = kb_prefs.get("main_chat_kb", "None")
    selected_kb = None if selected_kb_name == "None" else selected_kb_name
    
    client = get_chat_client("main_chat", selected_kb=selected_kb)
    
    if not client:
        return "I apologize, but I'm unable to connect to the AI service at the moment. Please try again later."
    
    try:
        # RAG: Combine uploaded files and knowledge base for comprehensive search
        relevant_chunks = []
        kb_name = selected_kb_name if selected_kb_name != "None" else None
        print(f"[GENERATE_AI] Selected KB: {kb_name}")
        print(f"\n[GENERATE_AI] Current chat_id: {st.session_state.get('current_chat_id')}")
        # Get chunks from uploaded files first (with higher priority)
        uploaded_chunks = []
        if st.session_state.current_chat_id:
            print(f"[GENERATE_AI] Checking uploaded files for chat_id: {st.session_state.current_chat_id}")
            uploaded_chunks = retrieve_relevant_chunks(user_text, st.session_state.current_chat_id)
            print(f"[GENERATE_AI] Retrieved {len(uploaded_chunks)} chunks from uploaded files")
            if uploaded_chunks:
                print(f"[GENERATE_AI] Uploaded file sources: {[c.get('filename') for c in uploaded_chunks]}")
        
        # Get chunks from knowledge base only if no strong uploaded file matches
        kb_chunks = []
        if kb_name and kb_name != "None":
            kb_chunks = retrieve_relevant_kb_chunks(user_text, kb_name)
            print(f"[GENERATE_AI] Retrieved {len(kb_chunks)} chunks from KB")
        
        # Prioritize uploaded files heavily when they exist
        if uploaded_chunks:
            # If we have uploaded files, use them as primary source
            print(f"[GENERATE_AI] Using uploaded files as primary source")
            # Get RAG settings to determine final top-k
            rag_settings = st.session_state.get("rag_settings", {})
            final_k = rag_settings.get("top_k_reranker", 20)
            
            relevant_chunks = uploaded_chunks[:final_k]  # Take up to final_k from uploaded files
            # Only add KB if uploaded files don't have enough content
            if len(uploaded_chunks) < final_k and kb_chunks:
                remaining_slots = final_k - len(uploaded_chunks)
                print(f"[GENERATE_AI] Supplementing with {min(remaining_slots, len(kb_chunks))} KB chunks")
                relevant_chunks.extend(kb_chunks[:remaining_slots])
        elif kb_chunks:
            # If no uploaded files, use knowledge base only
            rag_settings = st.session_state.get("rag_settings", {})
            final_k = rag_settings.get("top_k_reranker", 20)
            relevant_chunks = kb_chunks[:final_k]  # Take up to final_k from KB
        else:
            print(f"[GENERATE_AI] No uploaded files and no KB selected - using general knowledge")
        
        # Get current model information
        current_model = get_current_model_info()
        
        # Get current knowledge base and custom system prompt from preferences
        kb_prefs = st.session_state.user_preferences.get("selected_knowledge_base", {})
        kb_name = kb_prefs.get("main_chat_kb", "None")
        custom_system_prompt = ""
        if kb_name and kb_name != "None":
            # Find the knowledge base in the knowledge bases list (they're the same as custom_knowledge_bases)
            kbs = st.session_state.get("knowledge_bases", [])
            kb = next((kb for kb in kbs if kb.get("name") == kb_name), None)
            if kb and kb.get("system_prompt"):
                custom_system_prompt = kb["system_prompt"]
            else:
                print(f"[GENERATE_AI] No custom system prompt found for KB: {kb_name}")
                if kb:
                    print(f"[GENERATE_AI] KB data: {kb}")
                else:
                    print(f"[GENERATE_AI] KB not found in list. Available KBs: {[k.get('name') for k in kbs]}")
        
        # Prepare conversation history
        messages = []
        
        # Add system message with context if we have relevant chunks
        if relevant_chunks:
            context = "\n\n".join([f"Document: {chunk['filename']}\nContent: {chunk['content']}" for chunk in relevant_chunks])
            
            if custom_system_prompt:
                # Use custom system prompt with context - let custom prompt take full control
                system_message = f"""{custom_system_prompt}

                Context:
                {context}"""
            else:
                # Check if we have uploaded files vs knowledge base content
                uploaded_files = []
                kb_files = []
                
                for chunk in relevant_chunks:
                    filename = chunk.get('filename', '')
                    # Simple heuristic: if it's in uploaded_chunks list, it's uploaded
                    if chunk in uploaded_chunks:
                        uploaded_files.append(filename)
                    else:
                        kb_files.append(filename)
                
                uploaded_files = list(set(uploaded_files))
                kb_files = list(set(kb_files))
                
                if uploaded_files and not kb_files:
                    # Only uploaded files
                    file_context = f"recently uploaded documents: {', '.join(uploaded_files)}"
                    priority_instruction = "Focus EXCLUSIVELY on the uploaded documents. The user just uploaded these files and wants information specifically about them."
                elif uploaded_files and kb_files:
                    # Both sources
                    file_context = f"recently uploaded documents ({', '.join(uploaded_files)}) and knowledge base ({', '.join(kb_files)})"
                    priority_instruction = "PRIORITIZE information from the uploaded documents as the user just uploaded them. Only supplement with knowledge base information if it directly relates to or enhances understanding of the uploaded content."
                else:
                    # Only knowledge base
                    file_context = f"Knowledge base"
                    priority_instruction = "Use the available knowledge base information to answer the question."
                
                # Enhanced system prompt with upload awareness
                system_message = f"""Help user with their query. 

IMPORTANT: You have access to {file_context}.

{priority_instruction}

When the user asks "what is this file" or similar questions about uploaded content, they are referring to the documents they just uploaded to this conversation, NOT to files in the knowledge base.

Context from sources:
{context}

Please provide helpful, accurate responses. Do not list the source files in your response as they will be displayed separately."""
            messages.append(SystemMessage(content=system_message))
        else:
            # When no knowledge base context, add general system message
            if custom_system_prompt:
                # Use custom system prompt for general chat - let custom prompt take full control
                system_message = custom_system_prompt
            else:
                # Use default system prompt
                system_message = f"""Help user with their query to the best of your ability using your general knowledge."""
            messages.append(SystemMessage(content=system_message))
        
        # Add conversation history (last 10 messages for context)
        for msg in st.session_state.messages[-10:]:
            if msg["role"] == "user":
                # Handle text messages (images are processed separately for current message)
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
            
        # Add current user message (text only)
        if relevant_chunks:
            # Safely extract content from chunks, handling different formats
            chunk_contents = []
            for chunk in relevant_chunks:
                if isinstance(chunk, dict):
                    if "content" in chunk:
                        chunk_contents.append(str(chunk["content"]))
                    elif "page_content" in chunk:
                        chunk_contents.append(str(chunk["page_content"]))
                    else:
                        # If it's a dict but no recognized content key, convert to string
                        chunk_contents.append(str(chunk))
                else:
                    # If it's not a dict, convert to string
                    chunk_contents.append(str(chunk))
            
            context_text = "\n\n".join(chunk_contents)
            final_user_message = f"Context from uploaded documents and knowledge base:\n{context_text}\n\nUser question: {user_text}"
        else:
            final_user_message = user_text
        
        # Check if we have images and if the model supports vision
        current_model_name = st.session_state.user_preferences.get("workflow_models", {}).get("main_chat", "")
        
        if uploaded_images and is_vision_capable_model(current_model_name):
            # Create vision message with images
            print(f"[GENERATE_AI] Creating vision message with {len(uploaded_images)} images for model {current_model_name}")
            vision_message_content = create_vision_message_content(final_user_message, uploaded_images)
            # Pass as dict to preserve complex content structure for vision models
            messages.append({"role": "user", "content": vision_message_content})
        else:
            # Create text-only message
            if uploaded_images:
                print(f"[GENERATE_AI] Images uploaded but model {current_model_name} is not vision-capable, using text only")
            messages.append(HumanMessage(content=final_user_message))

        # Get response
        response = client.invoke(messages)
        
        # Clear uploaded images after processing
        if uploaded_images:
            st.session_state.uploaded_images = []
            print(f"[GENERATE_AI] Cleared {len(uploaded_images)} uploaded images after processing")
        
        # Store relevant chunks for display
        if relevant_chunks:
            st.session_state.temp_sources = relevant_chunks
        
        return response.content
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"

def retrieve_relevant_chunks(query, chat_id, k=None):
    """Retrieve relevant chunks from chat embeddings"""
    try:
        from utils.vector_db import load_chat_embeddings
        import numpy as np
        import faiss
        
        # Get k from RAG settings if not provided
        if k is None:
            rag_settings = st.session_state.get("rag_settings", {})
            reranking_enabled = rag_settings.get("reranking_engine", False)
            if reranking_enabled:
                k = rag_settings.get("top_k_reranker", 5)
            else:
                k = rag_settings.get("top_k", 5)
        
        if not LLM_AVAILABLE:
            print(f"[RETRIEVE_CHUNKS] LLM not available")
            return []
                
        # Load chat embeddings
        index, metadata = load_chat_embeddings(chat_id)
        if index is None or metadata is None or not metadata.get("chunks"):
            return []
        
        print(f"[RETRIEVE_CHUNKS] Found {len(metadata.get('chunks', []))} chunks in embeddings")
        print(f"[RETRIEVE_CHUNKS] Filenames: {metadata.get('filenames', [])}")
        
        # Get embedding for the query
        embedding_client = get_embedding_client()
        query_embedding = embedding_client.embed_documents([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = index.search(query_embedding, min(k, index.ntotal))
        
        print(f"[RETRIEVE_CHUNKS] Search scores: {scores[0]}")
        print(f"[RETRIEVE_CHUNKS] Search indices: {indices[0]}")
        
        # Prepare results - no threshold to guarantee top-k chunks
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metadata["chunks"]):  # No threshold - return all top-k
                relevant_chunks.append({
                    "content": metadata["chunks"][idx],
                    "filename": metadata["filenames"][idx],
                    "score": float(score)
                })
                print(f"[RETRIEVE_CHUNKS] Added chunk with score {score:.3f} from {metadata['filenames'][idx]}")
        
        print(f"[RETRIEVE_CHUNKS] Returning {len(relevant_chunks)} relevant chunks")
        return relevant_chunks
    except Exception as e:
        print(f"[RETRIEVE_CHUNKS] Error retrieving chunks: {e}")
        import traceback
        traceback.print_exc()
        return []

def retrieve_relevant_kb_chunks(query, kb_name, k=None):
    """Retrieve relevant chunks from the selected knowledge base"""
    try:        
        # Get RAG settings first to determine k
        rag_settings = st.session_state.get("rag_settings", {})
        reranking_enabled = rag_settings.get("reranking_engine", False)
        
        # Use top_k_reranker if reranking is enabled, otherwise use top_k
        if k is None:
            if reranking_enabled:
                k = rag_settings.get("top_k_reranker", 5)
            else:
                k = rag_settings.get("top_k", 5)
                
        # Find the KB metadata from session state
        kb_list = st.session_state.get("knowledge_bases", [])
        kb = next((kb for kb in kb_list if kb.get("name") == kb_name), None)
        if not kb:
            print(f"[KB_RAG] KB '{kb_name}' not found in knowledge_bases list")
            return []
        
        # Get KB ID for vector database access
        kb_id = kb.get('id')
        if not kb_id:
            print(f"[KB_RAG] KB '{kb_name}' has no ID - cannot access vector database")
            return []
            
        print(f"[KB_RAG] Using KB ID: '{kb_id}' for KB: '{kb_name}'")
        
        # Load vector database using ID-based system
        from utils.vector_db import load_vector_db
        import numpy as np
        import faiss
        
        if not LLM_AVAILABLE:
            print(f"[KB_RAG] LLM not available")
            return []
        embedding_client = get_embedding_client()
        
        index, metadata = load_vector_db(kb_id)
        
        if index is None or not metadata:
            print(f"[KB_RAG] No vector database found for KB ID: {kb_id}")
            return []
        
        # Get embedding for the query
        query_embedding = embedding_client.embed_documents([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Get additional RAG settings
        top_k_reranker = rag_settings.get("top_k_reranker", k)
        
        # If reranking is enabled, get more candidates for reranking
        search_k = k * 2 if reranking_enabled else k
        search_k = min(search_k, index.ntotal)
        
        print(f"[KB_RAG] Reranking enabled: {reranking_enabled}, searching top {search_k}, will rerank to top {top_k_reranker}")

        # Search for similar chunks
        scores, indices = index.search(query_embedding, search_k)

        # Prepare initial results - no threshold to guarantee top-k chunks
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(metadata["chunks"]):  # No threshold - return all top-k
                relevant_chunks.append({
                    "content": metadata["chunks"][idx],
                    "filename": metadata["filenames"][idx],
                    "score": float(score)
                })
        
        # Apply reranking if enabled and we have multiple chunks
        if reranking_enabled and len(relevant_chunks) > 1:
            relevant_chunks = apply_reranking_to_chunks(query, relevant_chunks, rag_settings)
            # Limit to top_k_reranker
            relevant_chunks = relevant_chunks[:top_k_reranker]
            print(f"[KB_RAG] After reranking, returning top {len(relevant_chunks)} chunks")
        else:
            # No reranking, just limit to k
            relevant_chunks = relevant_chunks[:k]
        
        return relevant_chunks
    except Exception as e:
        print(f"[KB_RAG] Error retrieving KB chunks: {e}")
        import traceback
        traceback.print_exc()
        return []

def apply_reranking_to_chunks(query, chunks, rag_settings):
    """Apply reranking algorithm to improve chunk relevance"""
    try:
        # Simple keyword overlap scoring for reranking
        query_words = set(query.lower().split())
        
        for chunk in chunks:
            # Safely extract chunk text content
            chunk_content = chunk.get("content", "")
            if isinstance(chunk_content, dict):
                # If chunk content is a dict, try to get content from common keys
                chunk_text = (chunk_content.get("content") or 
                             chunk_content.get("page_content") or 
                             str(chunk_content))
            elif isinstance(chunk_content, str):
                chunk_text = chunk_content
            else:
                chunk_text = str(chunk_content) if chunk_content else ""
            
            content_words = set(chunk_text.lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            
            # Combine original score with keyword overlap using bm25_weight
            bm25_weight = rag_settings.get("bm25_weight", 0.5)
            chunk["rerank_score"] = (1 - bm25_weight) * chunk["score"] + bm25_weight * keyword_overlap
            
        # Sort by reranked score (highest first)
        chunks.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        
        # Update score to use reranked score for display
        for chunk in chunks:
            chunk["original_score"] = chunk["score"]
            chunk["score"] = chunk.get("rerank_score", chunk["score"])
        
        return chunks
        
    except Exception as e:
        print(f"[KB_RAG] Error in reranking: {e}")
        return chunks

def save_current_conversation():
    """Save the current conversation to the conversations list"""
    if st.session_state.current_conversation and st.session_state.messages:
        # Check if conversation already exists
        existing_index = None
        for i, conv in enumerate(st.session_state.conversations):
            if conv.get('title') == st.session_state.current_conversation:
                existing_index = i
                break
        
        conversation_data = {
            'title': st.session_state.current_conversation,
            'messages': st.session_state.messages,
            'date': datetime.now().strftime('%b %d, %I:%M %p'),
            'timestamp': datetime.now().isoformat(),
            'chat_id': st.session_state.current_chat_id,
            'type': 'chat'
        }
        
        if existing_index is not None:
            # Remove existing conversation and add updated one at the end (most recent)
            st.session_state.conversations.pop(existing_index)
            st.session_state.conversations.append(conversation_data)
        else:
            st.session_state.conversations.append(conversation_data)
        
        save_conversations(st.session_state.conversations)

# Add extra spacing (margin-bottom) below the chat box at the bottom of the ChatJPL conversation page.
st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()