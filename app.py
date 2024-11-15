import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain.prompts import ChatPromptTemplate
from typing import Generator
import time

st.set_page_config(layout="centered", page_icon="👔", page_title="Bedrock with RAG")


CHROMA_PATH = "db"
aws_profile_name = "aws-profile-name"
region_name = "region-name"
emb_model_name = "amazon.titan-embed-text-v2:0"
txt_model_name = "amazon.titan-text-express-v1"


if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_embedding_function():
    return BedrockEmbeddings(
        credentials_profile_name=aws_profile_name,
        region_name=region_name,
        model_id=emb_model_name,
    )


@st.cache_resource
def get_llm():
    return BedrockLLM(
        credentials_profile_name=aws_profile_name,
        region_name=region_name,
        model_id=txt_model_name,
        model_kwargs={
            "temperature": 0.2,
            "topP": 0.8,
            "maxTokenCount": 2048,
        },  # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
    )


@st.cache_resource
def init_chroma():
    return Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )


def stream_response(response: str) -> Generator[str, None, None]:
    """Simulate streaming by yielding response chunks"""
    words = response.split()
    for i in range(0, len(words), 3):
        chunk = " ".join(words[i : i + 3])
        yield chunk + " "
        time.sleep(0.05)


def query_documents(query_text):
    db = init_chroma()
    results = db.similarity_search_with_score(query_text, k=5)

    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=query_text
    )

    response = get_llm().invoke(prompt)
    sources = [doc.metadata["id"] for doc, _score in results]
    sources_unique = [c.split("..\\pdfs\\")[-1].split(".pdf")[0] for c in sources]
    return response, sources, list(set(sources_unique))


PROMPT_TEMPLATE = """
You are a researcher working on multimodal methods using multimodal data.
Answer the question based only on the following context:

{context}

---

Question: {question}
"""


st.markdown(
    """
### About this AI Assistant
This tool helps researchers explore multimodal and deep learning research:
* 📚 Access to research papers and documentation
* 💡 Answer questions about research
* 🔍 Search through source materials
* 📊 Cite relevant sources
"""
)

with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown(
        """
    **Model**: AWS Bedrock Titan \n
    **Database**: Chroma Vector DB \n
    **Sources**: Multimodal Research Papers \n
    """
    )
    st.markdown("---")
    st.markdown("Powered by LangChain + AWS Bedrock")

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

db = init_chroma()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("What would you like to know about multimodal research?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching for answer..."):
            response, sources, sources_unique = query_documents(query)

            response_placeholder = st.empty()
            displayed_response = ""

            for chunk in stream_response(response):
                displayed_response += chunk
                response_placeholder.markdown(displayed_response + "▌")

            full_response = f"{response}\n\n**Sources:**\n" + "\n".join(
                [f"- {s}" for s in sources_unique]
            )
            response_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
