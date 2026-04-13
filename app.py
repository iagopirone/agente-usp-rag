import streamlit as st
import pandas as pd
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =================================================================
# 1. CONFIGURAÇÃO DE AMBIENTE E INTERFACE
# =================================================================

# O código vai puxar a chave oculta do servidor do Streamlit
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Agente USP São Carlos", page_icon="🎓", layout="centered")

st.title("🎓 Agente Institucional - USP São Carlos")
st.markdown("""
Assistente de Inteligência Artificial alimentado pelos dados estruturados institucionais do 
campus de São Carlos da Universidade de São Paulo.
""")

# =================================================================
# 2. MOTOR DE INTELIGÊNCIA (PIPELINE RAG)
# =================================================================

@st.cache_resource
def configurar_agente():
    """
    Função para ler os CSVs, gerar embeddings locais e configurar o LLM.
    O uso de cache_resource evita que o sistema re-indexe os dados a cada clique.
    """
    # A. Leitura e Preparação dos Dados
    textos_combinados = []
    arquivos_csv = glob.glob("dados_2024/*.csv")
    
    if not arquivos_csv:
        return None, None
        
    for arquivo in arquivos_csv:
        nome_tabela = os.path.basename(arquivo).replace('.csv', '')
        df = pd.read_csv(arquivo)
        
        # Converte cada linha da planilha em um documento de texto para o Agente
        for index, row in df.iterrows(): 
            linha_texto = f"[Fonte: {nome_tabela}] " + ", ".join([f"{col}: {val}" for col, val in row.items()])
            textos_combinados.append(linha_texto)
            
    # B. Embeddings Locais (HuggingFace)
    # Modelo 'all-MiniLM-L6-v2': Rápido, leve e roda 100% no processador
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # C. Banco de Dados Vetorial (ChromaDB)
    # Armazena os textos e permite busca por similaridade semântica.
    vectorstore = Chroma.from_texts(texts=textos_combinados, embedding=embeddings)
    
    # D. Configuração do LLM   
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1)
    return vectorstore, llm

# Inicialização do Sistema
with st.spinner("Indexando base de dados institucional..."):
    vectorstore, llm = configurar_agente()

# =================================================================
# 3. INTERFACE DE CONSULTA
# =================================================================

if vectorstore and llm:
    pergunta = st.text_input("Digite sua dúvida sobre indicadores da USP São Carlos:", 
                            placeholder="Ex: Quantos docentes existem no ICMC e na EESC?")

    if st.button("Consultar Agente"):
        if pergunta:
            with st.spinner("Consultando tabelas e sintetizando resposta..."):
                # Busca os 30 trechos mais relevantes para garantir contexto completo
                documentos = vectorstore.similarity_search(pergunta, k=100)
                contexto = "\n".join([doc.page_content for doc in documentos])
                
                # Engenharia de Prompt: Instruções para comportamento de Analista
                prompt = f"""Você é um analista de dados especialista na USP de São Carlos.
                Você recebeu um contexto com dados de várias unidades (EESC, ICMC, IFSC, IQSC, IAU).
                
                Sua tarefa:
                1. Procure no contexto abaixo os dados de CADA unidade mencionada.
                2. Se a pergunta pede uma comparação, você DEVE listar os números de todas as unidades que encontrar.
                3. Não resuma a resposta se houver dados disponíveis para mais de uma unidade.
                4. Seja específico: cite o número de Graduação e o de Pós-Graduação para cada instituto.

                CONTEXTO DAS TABELAS:
                {contexto}
                
                PERGUNTA: {pergunta}
                RESPOSTA:"""
                
                # Execução da Cadeia
                try:
                    resposta = llm.invoke(prompt)
                    st.markdown("---")
                    st.subheader("💡 Resposta do Agente:")
                    st.write(resposta.content)
                except Exception as e:
                    st.error(f"Erro na comunicação com o cérebro da IA: {e}")
        else:
            st.warning("Por favor, insira uma pergunta antes de consultar.")
else:
    st.error("Erro ao carregar dados. Verifique se a pasta 'dados_2024' contém os arquivos CSV.")

# Rodapé informativo
st.markdown("---")
st.caption("Desenvolvido por Iago de Oliveira Pirone")
