# importing libaries
import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

st.title("News Research Tool")

st.sidebar.title("News Articles URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


main_placeholder = st.empty()
process_url_clicked = st.sidebar.button("Process URLs")
llm = ChatOllama(model="gemma3:1b")

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # create embeddings and save it to FAISS index
    embeddings = OllamaEmbeddings(model='mxbai-embed-large:latest')
    vectorstore_db = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    save_directory = "my_faiss_index"
    vectorstore_db.save_local(save_directory)

    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(save_directory):
            # load vector db
            loaded_vector_db = FAISS.load_local(
                save_directory, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=loaded_vector_db.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                # Split the sources by newline
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
