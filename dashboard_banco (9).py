import streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import squarify # Importa a biblioteca squarify para criar o mapa de árvore
import matplotlib.colors as mcolors # Para gerar uma paleta de cores

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency 

# --- Configurações da Página ---
st.set_page_config(
    page_title="Avaliação das Características Sociais, Educacionais e de Saúde das Pessoas com Síndrome de Down no Brasil",
    page_icon="🧠",
    layout="wide"
)

# --- Função de Carregamento de Dados (com cache) ---
@st.cache_data
def load_data():
    try:
        # Please adjust this path
        #df = pd.read_excel('C:/Users/ellen/Downloads/Banco_SD.xlsx')
        df = pd.read_excel('C:/Users/Emille/Documents/UNIFESP/MATÉRIAS/Tópicos em Ciência de Dados para Neurociência/Projeto5/Banco_SD.xlsx')
        return df
    except FileNotFoundError:
        st.error("Erro: O arquivo 'Banco_SD.xlsx' não foi encontrado.")
        st.info("Por favor, verifique se o nome do arquivo e o caminho estão corretos.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados: {e}")
        st.stop()

# --- Aplica estilos CSS globais ---
st.markdown("""
<style>
.justified-text {
    text-align: justify;
}
.variable-list-container ul {
    list-style-type: disc;
    padding-left: 20px;
}
.variable-list-container li {
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# --- Função principal que cria o dashboard ---
def create_dashboard(df_original):
    if not df_original.empty:
        st.title("Avaliação das Características Sociais, Educacionais e de Saúde das Pessoas com Síndrome de Down no Brasil")

        # --- Abas com títulos mais curtos ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "1. Contexto",             # Shortened
            "2. Dados",                # Shortened
            "3. Sociodemográficas",
            "4. Educacionais",
            "5. Saúde e Estilo de Vida",         # Shortened
            "6. Participação, Autonomia e Social",  # Shortened
            "7. Correlações",
            "8. PCA e Cluster",
            "9. Modelos ML",           # Shortened (Machine Learning Models)
            "10. Discussão"            # Shortened
        ])

        # --- Sidebar para filtros ---
        df_filtrado = df_original.copy()


        # --- Sidebar com Sumário das Abas ---
        st.sidebar.header("📋 Sumário das Abas")
        
        with st.sidebar.expander("1️⃣ Contexto e Objetivo"):
            st.header("Contexto do Estudo")
            st.subheader("- Estudo sobre Avaliação das Características Sociais, Educacionais e de Saúde das Pessoas com Síndrome de Down no Brasil")
            st.subheader("- Síndrome de Down")
            st.subheader("- Panorama no Brasil")
            st.subheader("- Objetivo")
        
        with st.sidebar.expander("2️⃣ Apresentação dos Dados"):
            st.header("Dados Gerais")
            st.subheader(f"- Total de participantes: {len(df_filtrado)}")
            st.subheader("- Período de coleta: [setembro de 2024 - atual]")
            st.subheader("- Regiões abrangidas: [18 Estados] AC, AL, BA, CE, ES, GO, MA, MT, MG, PB, PR, PE, PI, RJ, RN, RS, SC e SP) mais o Distrito Federal. ")
            
            st.header("Variáveis Principais")
            st.subheader("- 159 colunas e 368 linhas")
            st.subheader("- Dados quantitativos e qualitativos")
            st.subheader("- Blocos temáticos de análise")
        
        with st.sidebar.expander("3️⃣ Sociodemográficas"):
            st.header("Principais Análises e Gráficos")
            st.subheader("- Distribuição por cor/etnia")
            st.subheader("- Quantos irmãos o participante possui")
            st.subheader("- Distribuição da renda familiar")
            st.subheader("- Recebe Bolsa família")
            st.subheader("- Recebe Benefício de Prestação Continuada")
            st.subheader("- Distribuição do tipo de moradia")
            st.subheader("- Distribuição por Faixa Etária")
        
        with st.sidebar.expander("4️⃣ Educacionais"):
            st.header("Principais Análises e Gráficos")
            st.subheader("- Nível de Escolaridade do Participante")
            st.subheader("- Nível de Escolaridade do Responsável")
            st.subheader("- O Participante é Alfabetizado?")
            st.subheader("- O Participante Sabe Ler?")
            st.subheader("- O Participante Sabe Escrever?")
            st.subheader("- O Participante consegue interpretar texto?")
            
        with st.sidebar.expander("5️⃣ Saúde e Estilo de Vida"):
            st.header("Principais Análises e Gráficos")
            st.subheader("- Distribuição do IMC por Sexo (%)")
            st.subheader("- Idade em que a mãe do participante teve a gestação")
            st.subheader("- Rede de Acompanhamento Pré-natal da Mãe")
            st.subheader("- Quando o Diagnóstico do Participante foi Realizado")
            st.subheader("- Foi feito o cariótipo?")
            st.subheader("- Irmãos do participante possuem alguma deficiência?")
            st.subheader("- Saúde Geral do Participante (Percepção do Cuidador)")
            st.subheader("- Rede de Acompanhamento Psicológico")
            st.subheader("- Rede de Estimulação Precoce")
            st.subheader("- Rede de Acompanhamento com Clínico Geral")
            st.subheader("- Rede de Acompanhamento com Dentista")
            st.subheader("- Rede de Acompanhamento com Nutricionista")
            st.subheader("- Rede de Acompanhamento com Oftalmologista")
            st.subheader("- Distribuição de Diagnósticos Psicológicos")
            st.subheader("- Distribuição de Imunizações (Vacinas)")
            st.subheader("- Prevalência de Morbidades e/ou Doenças")
            st.subheader("- Doenças Apesar das Imunizações")
            st.subheader("- Gravidade da COVID")
            st.subheader("- Distribuição de Doses de Vacina COVID-19")
            st.subheader("- Medicações Utilizadas pelos Participantes")
            st.subheader("- Prática de Atividade Física")
            st.subheader("- Frequência Semanal de Atividade Física")
            st.subheader("- Hábitos Alimentares Saudáveis")
            st.subheader("- Atendimento de Saúde Público Adequado")
        
        with st.sidebar.expander("6️⃣ Participação, Autonomia e Social"):
            st.header("Principais Análises e Gráficos")
            st.subheader("- Autonomia para Tomar Decisões")
            st.subheader("- Deslocamento Independente")
            st.subheader("- Realização de Necessidades Pessoais")
            st.subheader("- Relacionamento Interpessoal")
            st.subheader("- Interação Social Voluntária")
            st.subheader("- Olhar do Cuidador Quanto a Tratamento com Respeito, Dignidade e Igualdade")
        
        with st.sidebar.expander("7️⃣ Correlações"):
            st.header("Correlações")
            st.subheader("- Correlação de Variáveis Socioeconômicas")
            st.subheader("- Correlação de Escolaridade e Alfabetização")
            st.subheader("- Correlação de Variáveis Demográficas e de Saúde")
        
        with st.sidebar.expander("8️⃣ PCA e Cluster"):
            st.header("Redução de dimensionalidade e clusterização")
            st.subheader("- Pré-precessamento de dados")
            st.subheader("- Visualização dos Clusters (PCA)")
            st.subheader("- Médias por Cluster")
            st.subheader("- Distribuição de Atividade Física por Cluster")
            st.subheader("- Estatísticas Detalhadas de Atividade Física por Cluster")
            st.subheader("- Percepção de Atendimento de Saúde por Cluster")
        
        with st.sidebar.expander("9️⃣ Regressão e Random Forest"):
            st.header("Técnicas Utilizadas")
            
            st.header("Considerações")
            st.subheader("- Limitações do estudo")
            st.subheader("- Sugestões para pesquisas futuras")

        with st.sidebar.expander("🔟 Discussão"):
            st.header("Discussão e Conslusão")
            st.subheader("- Referências")
            st.subheader("- Recomendações políticas")
            
            st.header("Considerações")
            st.subheader("- Limitações do estudo")
            st.subheader("- Sugestões para pesquisas futuras")
            
        # --- CONTEÚDO DA ABA 1: Contexto e Objetivo ---
        with tab1:
            st.header("Contexto do Estudo")
            st.image("Image1.jpg", width=1000)
            st.image("Image2.jpg", width=1000)

        # --- CONTEÚDO DA ABA 2: Apresentação dos Dados ---
        with tab2:
            st.header("Apresentação dos Dados")

            st.markdown("""
            <div class="justified-text">
            O banco possui variáveis quantitativas e qualitativas com dados socioeconômico, demográfico, educacional e de saúde. O tipo de variável poderá ser visto no documento compartilhado em uma planilha no formato xlsx onde consta os dados e o dicionário.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="justified-text">
            Além disso, o banco possui 159 colunas e 368 linhas (contando com a primeira coluna e linha que são identificação (ID) e perguntas, respectivamente). Detalhes do banco podem ser vistos na planilha em formato xlsx.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="justified-text">
            As variáveis selecioandas para análise descritiva serão apresentadas nas abas (4 blocos) seguintes. Cada aba apresenta um "bloco" de variáveis agrupadas com características semelhantes: "Características Sociodemográficas", "Características Educacionais", "Características de Saúde e Estilo de Vida" e "Características de Participação, Autonomia e Interação Social".s
            </div>
            """, unsafe_allow_html=True)

            st.header("Variáveis")

            st.markdown("""
            <div class="variable-list-container">
                <ul>
                    <li>ID</li>
                    <li>Instituição</li>
                    <li>Data de nascimento</li>
                    <li>Idade do participante</li>
                    <li>Cidade</li>
                    <li>Naturalidade</li>
                    <li>Cuidador principal</li>
                    <li>Idade do cuidador principal</li>
                    <li>Peso do participante</li>
                    <li>IMC</li>
                    <li>Classificação do IMC</li>
                    <li>Peso do cuidador principal</li>
                    <li>Altura</li>
                    <li>Altura do cuidador principal</li>
                    <li>Sexo do participante</li>
                    <li>Cor/etnia do participante</li>
                    <li>Possui alguma deficiência</li>
                    <li>Se na pergunta anterior for selecionado outra DI, escreve qual</li>
                    <li>CID do participante</li>
                    <li>O participante foi adotado</li>
                    <li>Idade que a mãe do participante teve a gestação</li>
                    <li>A mãe do participante realizou o acompanhamento pré-natal</li>
                    <li>Realizou o acompanhamento pré-natal na rede</li>
                    <li>O diagnóstico do participante foi realizado no</li>
                    <li>Foi feito o cariótipo</li>
                    <li>O participante possui irmãos</li>
                    <li>Quantos irmãos o participante possui</li>
                    <li>Os irmãos do participante possuem alguma deficiência</li>
                    <li>Você considera que a saúde do participante em geral é</li>
                    <li>O participante realiza/realizou acompanhamento psicológico</li>
                    <li>Realiza/realizou na rede</li>
                    <li>Realizou algum programa de intervenção/estimulação precoce</li>
                    <li>Realiza/realizou na rede</li>
                    <li>O participante faz acompanhamento com clínico geral</li>
                    <li>Se sim, o acompanhamento ocorre na rede</li>
                    <li>O participante faz acompanhamento com dentista</li>
                    <li>Se sim, o acompanhamento ocorre na rede</li>
                    <li>O participante faz acompanhamento com nutricionista</li>
                    <li>Se sim, o acompanhamento ocorre na rede</li>
                    <li>O participante faz acompanhamento com oftalmologista</li>
                    <li>Se sim, o acompanhamento ocorre na rede</li>
                    <li>Apresenta alguma das morbidades e/ou doenças a seguir</li>
                    <li>Recebeu alguma das imunizações a seguir</li>
                    <li>Apesar das imunizações, teve alguma dessas doenças</li>
                    <li>Teve Covid-19</li>
                    <li>Gravidade da COVID do participante</li>
                    <li>O participante ficou hospitalizado (na UTI) por conta da COVID</li>
                    <li>O participante precisou de suplementação de O2 por conta da COVID</li>
                    <li>Tomou a 1ª dose da COVID</li>
                    <li>Nome da vacina da 1ª dose da COVID</li>
                    <li>Data da 1ª dose da COVID</li>
                    <li>Tomou a 2ª dose da COVID</li>
                    <li>Nome da vacina da 2ª dose da COVID</li>
                    <li>Data da 2ª dose da COVID</li>
                    <li>Tomou a 3ª dose da COVID</li>
                    <li>Nome da vacina da 3ª dose da COVID</li>
                    <li>Data da 3ª dose da COVID</li>
                    <li>Tomou a 4ª dose da COVID</li>
                    <li>Nome da vacina da 4ª dose da COVID</li>
                    <li>Data da 4ª dose da COVID</li>
                    <li>Tomou a 5ª dose da COVID</li>
                    <li>Nome da vacina da 5ª dose da COVID</li>
                    <li>Data da 5ª dose da COVID</li>
                    <li>Tomou a 6ª dose da COVID</li>
                    <li>Nome da vacina da 6ª dose da COVID</li>
                    <li>Data da 6ª dose da COVID</li>
                    <li>O participante toma alguma dessas medicações</li>
                    <li>O participante é alfabetizado</li>
                    <li>Sabe ler</li>
                    <li>Sabe escrever</li>
                    <li>O participante consegue interpretar texto</li>
                    <li>O participante lê jornais, revista ou livros</li>
                    <li>Nível de escolaridade do participante</li>
                    <li>Nível de escolaridade do responsável do participante</li>
                    <li>Renda familiar</li>
                    <li>Quantas pessoas vivem com esta renda familiar</li>
                    <li>Recebe Bolsa Família</li>
                    <li>Recebe BPC</li>
                    <li>A residência é</li>
                    <li>Pratica atividade física ou esporte</li>
                    <li>Qual atividade física o participante pratica</li>
                    <li>O participante pratica atividade física ou esporte quantas vezes na semana</li>
                    <li>Por quanto tempo o participante pratica essa atividade física e/ou esporte</li>
                    <li>Você considera que o participante possui autonomia para tomar decisões e tem a chance de escolher as coisas que deseja (escolher as coisas para a sua satisfação pessoal)</li>
                    <li>O participante se desloca independentemente pela cidade</li>
                    <li>O participante realiza alguma atividade artística</li>
                    <li>Na maioria das vezes, você considera o participante na realização de suas necessidades pessoais como</li>
                    <li>Você considera que o participante se relaciona com diferentes pessoas, tem amigos e se dá bem com as pessoas</li>
                    <li>Você considera que o participante, por vontade própria interage socialmente, frequentando diferentes lugares da cidade ou do bairro</li>
                    <li>Você considera que o participante é tratado com respeito, dignidade e igualdade pelas outras pessoas</li>
                    <li>Você considera que o participante possui hábitos alimentares saudáveis</li>
                    <li>Você considera que o participante possui atendimento de saúde adequado pelo sistema público para atender suas necessidades</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- ABA 3: Características Sociodemográficas (Gráficos Matplotlib) ---
        with tab3:
            st.header("Características Sociodemográficas")

            # Primeira linha de gráficos
            col1_row1, col2_row1 = st.columns(2)

            with col1_row1:
                st.subheader("Distribuição por Cor/Etnia")
                if 'Cor/etnia do participante' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Cor/etnia do participante'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(7, 6)) # Ajuste no figsize para proporção
                    bars = ax.bar(contagem.index, contagem.values, color='#bd928b', edgecolor='#bd928b')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(i, valor + (total * 0.01), f'{percentual:.1f}%', ha='center', va='bottom', fontsize=9)

                    ax.set_title('Distribuição por Cor/Etnia', fontsize=12)
                    ax.set_xlabel('Cor/Etnia do Participante', fontsize=10)
                    ax.set_ylabel('Frequência', fontsize=10)
                    ax.tick_params(axis='x', rotation=45, labelsize=9)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values(): # Manter bordas visíveis
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True) # Adicionado use_container_width
                    plt.close(fig)
                else:
                    st.warning("Não foi possível gerar o gráfico de Cor/Etnia. Verifique se a coluna existe ou se o DataFrame está vazio após os filtros.")

            with col2_row1:
                st.subheader("Quantos irmãos o participante possui")
                if 'Quantos irmãos o participante possui' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Quantos irmãos o participante possui'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(7, 6)) # Ajuste no figsize para proporção
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#a1a6aa', edgecolor='#a1a6aa')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + (contagem.max() * 0.05), i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)

                    ax.set_title('Quantos irmãos o participante possui', fontsize=12)
                    ax.set_xlabel('Frequência', fontsize=10)
                    ax.set_ylabel('Nº de irmãos', fontsize=10)
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    for spine in ax.spines.values(): # Manter bordas visíveis
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True) # Adicionado use_container_width
                    plt.close(fig)
                else:
                    st.warning("Não foi possível gerar o gráfico de 'Quantos irmãos o participante possui'. Verifique a coluna.")


            # Segunda linha de gráficos
            col1_row2, col2_row2 = st.columns(2)

            with col1_row2:
                st.subheader("Distribuição da renda familiar")
                if 'Renda familiar' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Renda familiar'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(7, 6)) # Ajuste no figsize para proporção
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#ffab03', edgecolor='#ffab03')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + (contagem.max() * 0.05), i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)

                    ax.set_title('Distribuição da renda familiar', fontsize=12)
                    ax.set_xlabel('Frequência', fontsize=10)
                    ax.set_ylabel('Renda', fontsize=10)
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    for spine in ax.spines.values(): # Manter bordas visíveis
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True) # Adicionado use_container_width
                    plt.close(fig)
                else:
                    st.warning("Não foi possível gerar o gráfico de 'Renda familiar'. Verifique a coluna.")

            with col2_row2:
                st.subheader("Recebe Bolsa família")
                if 'Recebe Bolsa família' in df_filtrado.columns and not df_filtrado.empty:
                    # Padronizar os valores na coluna
                    df_filtrado['Recebe Bolsa família'] = (
                        df_filtrado['Recebe Bolsa família']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['Recebe Bolsa família'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(7, 6)) # Ajuste no figsize para proporção
                    bars = ax.bar(labels_formatados, contagem.values, color='#e2aa87', edgecolor='#e2aa87')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + (total * 0.01), f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Recebe Bolsa família', fontsize=12)
                    ax.set_xlabel('Resposta', fontsize=10)
                    ax.set_ylabel('Frequência', fontsize=10)
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    for spine in ax.spines.values(): # Manter bordas visíveis
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True) # Adicionado use_container_width
                    plt.close(fig)
                else:
                    st.warning("Não foi possível gerar o gráfico de 'Recebe Bolsa família'. Verifique a coluna.")

            # Terceira linha de gráficos
            col1_row3, col2_row3 = st.columns(2)

            with col1_row3:
                st.subheader("Recebe Benefício de Prestação Continuada")
                if 'Recebe BPC' in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['Recebe BPC'] = (
                        df_filtrado['Recebe BPC']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['Recebe BPC'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(7, 6)) # Ajuste no figsize para proporção
                    bars = ax.bar(labels_formatados, contagem.values, color='#b6d4bb', edgecolor='#b6d4bb')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + (total * 0.01), f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Recebe Benefício de Prestação Continuada', fontsize=12)
                    ax.set_xlabel('Resposta', fontsize=10)
                    ax.set_ylabel('Frequência', fontsize=10)
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    for spine in ax.spines.values(): # Manter bordas visíveis
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True) # Adicionado use_container_width
                    plt.close(fig)
                else:
                    st.warning("Não foi possível gerar o gráfico de 'Recebe BPC'. Verifique a coluna.")

            with col2_row3:
                st.subheader("Distribuição do tipo de moradia")
                if 'A residência é' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['A residência é'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(7, 6)) # Ajuste no figsize para proporção
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#a4d9a3', edgecolor='#a4d9a3')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + (contagem.max() * 0.05), i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)

                    ax.set_title('Distribuição do tipo de moradia', fontsize=12)
                    ax.set_xlabel('Frequência', fontsize=10)
                    ax.set_ylabel('Moradia', fontsize=10)
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    for spine in ax.spines.values(): # Manter bordas visíveis
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True) # Adicionado use_container_width
                    plt.close(fig)
                else:
                    st.warning("Não foi possível gerar o gráfico de 'A residência é'. Verifique a coluna.")
                    
            # Quarta linha de gráficos
            col1_row4, col2_row4 = st.columns(2)
            # --- GRÁFICO DE DISTRIBUIÇÃO POR FAIXA ETÁRIA (IDÊNTICO À ANÁLISE ORIGINAL) ---
            with col1_row4:  # Segunda coluna da primeira linha
                st.subheader("Distribuição por Faixa Etária")
                
                if 'Idade do participante' in df_filtrado.columns and not df_filtrado.empty:
                    # CÓPIA FIEL DO PROCESSAMENTO DO SEU GRÁFICO SIMPLES
                    df_filtrado['Idade_num'] = pd.to_numeric(df_filtrado['Idade do participante'], errors='coerce')
                    
                    # USAR EXATAMENTE OS MESMOS PARÂMETROS DO SEU CÓDIGO ORIGINAL
                    bins = [0, 10, 19, 29, 39, 49, 59]
                    labels = ['5 meses a 10 anos', '11 anos a 19 anos', '20 anos a 29 anos',
                             '30 anos a 39 anos', '40 anos a 49 anos', '50 anos a 59 anos']
                    
                    # MESMA LÓGICA DE CLASSIFICAÇÃO (com right=True)
                    df_filtrado['Faixa Etária'] = pd.cut(
                        df_filtrado['Idade_num'],
                        bins=bins,
                        labels=labels,
                        right=True  # INCLUSÃO DO LIMITE SUPERIOR (igual ao seu código)
                    )
                    
                    # MESMA CONTAGEM E ORDENAÇÃO
                    contagem = df_filtrado['Faixa Etária'].value_counts().sort_index()
                    total = contagem.sum()
            
                    # GRÁFICO COM MESMO ESTILO VISUAL
                    fig, ax = plt.subplots(figsize=(7, 5))
                    bars = ax.bar(
                        contagem.index,
                        contagem.values,
                        color='#bd928b',
                        edgecolor='#bd928b',
                        width=0.8
                    )
            
                    # MESMO CÁLCULO DE PORCENTAGENS
                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(
                            i,
                            valor + (total * 0.01),
                            f'{percentual:.1f}%',
                            ha='center',
                            va='bottom',
                            fontsize=10
                        )
                    # Configurações do eixo X para alinhamento perfeito
                    ax.set_xticks(range(len(contagem)))  # Define um tick para cada barra
                    ax.set_xticklabels(contagem.index, rotation=45, ha='right', fontsize=10)  # ha='right' para melhor alinhamento
                    
                    # Configurações do gráfico
                    ax.set_title('Distribuição por Faixa Etária', fontsize=14, pad=20)
                    ax.set_xlabel('Faixa Etária', fontsize=12)
                    ax.set_ylabel('Número de Participantes', fontsize=12)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    ax.set_ylim(0, contagem.max() * 1.15)  # Espaço para os rótulos
            
                    # MESMA FORMATAÇÃO DE EIXOS
                    #ax.set_title('Distribuição por Faixa Etária', fontsize=12)
                    #ax.set_xlabel('Faixa Etária', fontsize=10)
                    #ax.set_ylabel('Número de Participantes', fontsize=10)
                    #ax.tick_params(axis='x', rotation=45, labelsize=9)
                    #ax.grid(axis='y', linestyle='--', alpha=0.5)
                    #ax.set_ylim(0, contagem.max() * 1.15)
            
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    
                else:
                    st.warning("Dados de idade não disponíveis ou DataFrame vazio após filtros.")
        
        # --- ABA 4: Características Educacionais (Novos Gráficos Matplotlib em Colunas) ---
        with tab4:
            st.header("Características Educacionais")

            # Linha 1 de gráficos Educacionais
            col1_edu_row1, col2_edu_row1 = st.columns(2)

            with col1_edu_row1:
                st.subheader("Nível de Escolaridade do Participante")
                if 'Nível de escolaridade do participante' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Nível de escolaridade do participante'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(8, 5)) # Mantido o figsize do seu código
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#7bbda1', edgecolor='#7bbda1')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição do nível de escolaridade do participante')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Escolaridade')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Nível de escolaridade do participante' não encontrada ou DataFrame vazio.")

            with col2_edu_row1:
                st.subheader("Nível de Escolaridade do Responsável")
                if 'Nível de escolaridade do responsável do participante' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Nível de escolaridade do responsável do participante'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(8, 5)) # Mantido o figsize do seu código
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#fcb653', edgecolor='#fcb653')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição do nível de escolaridade do responsável do participante')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Escolaridade')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Nível de escolaridade do responsável do participante' não encontrada ou DataFrame vazio.")

            # Linha 2 de gráficos Educacionais
            col1_edu_row2, col2_edu_row2 = st.columns(2)

            with col1_edu_row2:
                st.subheader("O Participante é Alfabetizado?")
                if 'O participante é alfabetizado' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['O participante é alfabetizado'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5)) # Mantido o figsize do seu código
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#cee879', edgecolor='#cee879')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição do nível de alfabetização do participante')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Alfabetização')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'O participante é alfabetizado' não encontrada ou DataFrame vazio.")

            with col2_edu_row2:
                st.subheader("O Participante Sabe Ler?")
                if 'Sabe ler' in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['Sabe ler'] = (
                        df_filtrado['Sabe ler']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['Sabe ler'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(6, 5)) # Mantido o figsize do seu código
                    bars = ax.bar(labels_formatados, contagem.values, color='#daa979', edgecolor='#daa979')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Distribuição dos participantes que sabem ler')
                    ax.set_xlabel('Resposta')
                    ax.set_ylabel('Frequência')
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Sabe ler' não encontrada ou DataFrame vazio.")

            # Linha 3 de gráficos Educacionais
            col1_edu_row3, col2_edu_row3 = st.columns(2)

            with col1_edu_row3:
                st.subheader("O Participante Sabe Escrever?")
                if 'Sabe escrever' in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['Sabe escrever'] = (
                        df_filtrado['Sabe escrever']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['Sabe escrever'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(6, 5)) # Mantido o figsize do seu código
                    bars = ax.bar(labels_formatados, contagem.values, color='#f9f36a', edgecolor='#f9f36a')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Distribuição dos participantes que sabem escrever')
                    ax.set_xlabel('Resposta')
                    ax.set_ylabel('Frequência')
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Sabe escrever' não encontrada ou DataFrame vazio.")

            with col2_edu_row3:
                st.subheader("O Participante Consegue Interpretar Texto?")
                if 'O participante consegue interpretar texto?' in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['O participante consegue interpretar texto?'] = (
                        df_filtrado['O participante consegue interpretar texto?']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['O participante consegue interpretar texto?'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(6, 5)) # Mantido o figsize do seu código
                    bars = ax.bar(labels_formatados, contagem.values, color='#88b4ec', edgecolor='#88b4ec')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Distribuição dos participantes que conseguem interpretar texto')
                    ax.set_xlabel('Resposta')
                    ax.set_ylabel('Frequência')
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'O participante consegue interpretar texto?' não encontrada ou DataFrame vazio.")


        # --- ABA 5: Características de Saúde e Estilo de Vida (Gráficos Matplotlib em Colunas) ---
        with tab5:
            st.header("Características de Saúde e Estilo de Vida")
        
            # Tabela de cruzamento entre Sexo e Classificação IMC
            st.subheader("Distribuição do IMC por Sexo (%)")
            
            # Processamento dos dados para a tabela
            if 'Sexo do participante' in df_filtrado.columns and 'Classificação IMC' in df_filtrado.columns and not df_filtrado.empty:
                # Garantir que a coluna 'IMC' é numérica
                df_filtrado['IMC'] = pd.to_numeric(df_filtrado['IMC'], errors='coerce')
                
                # Definir condições e classificações
                condicoes = [
                    (df_filtrado['IMC'] < 18.5),
                    (df_filtrado['IMC'] >= 18.5) & (df_filtrado['IMC'] <= 24.9),
                    (df_filtrado['IMC'] >= 25) & (df_filtrado['IMC'] <= 29.9),
                    (df_filtrado['IMC'] >= 30) & (df_filtrado['IMC'] <= 34.9),
                    (df_filtrado['IMC'] >= 35) & (df_filtrado['IMC'] <= 39.9),
                    (df_filtrado['IMC'] >= 40)
                ]
                escolhas = [
                    'Abaixo do peso',
                    'Normal',
                    'Sobrepeso',
                    'Obesidade Grau I',
                    'Obesidade Grau II',
                    'Obesidade Grau III'
                ]
                
                # Calcular classificação
                classificacao_calculada = np.select(condicoes, escolhas, default=pd.NA)
                df_filtrado['Classificação IMC'] = df_filtrado['Classificação IMC'].fillna(
                    pd.Series(classificacao_calculada, index=df_filtrado.index))
                df_filtrado['Classificação IMC'] = df_filtrado['Classificação IMC'].str.strip()
                
                # Criar tabela de cruzamento
                tabela_cruzada = pd.crosstab(
                    df_filtrado['Sexo do participante'], 
                    df_filtrado['Classificação IMC'], 
                    normalize='index') * 100
                
                # Exibir tabela formatada
                st.dataframe(tabela_cruzada.style.format("{:.1f}%"), use_container_width=True)
            else:
                st.warning("Colunas necessárias para a tabela não encontradas ou DataFrame vazio.")

            # Linha 2 de gráficos de Saúde
            col1_saude_row2, col2_saude_row2 = st.columns(2)

            with col1_saude_row2:
                # Linha 1 de gráficos de Saúde (agora apenas com o gráfico de idade gestacional)
                st.subheader("Idade em que a mãe do participante teve a gestação")
                nome_coluna_idade_mae_original = 'Idade em que a mãe do participante teve a gestação'
                if nome_coluna_idade_mae_original in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['Idade em que a mãe do participante teve a gestação_Numerica'] = pd.to_numeric(
                        df_filtrado[nome_coluna_idade_mae_original], errors='coerce'
                    )
                    bins = [15, 24, 30, 35, 40, 45, 48]
                    labels = ['16 - 24', '25 - 30', '31 - 35', '36 - 40', '41 - 45', '46 - 48']
                    df_filtrado['Faixa Etária da Mãe'] = pd.cut(
                        df_filtrado['Idade em que a mãe do participante teve a gestação_Numerica'],
                        bins=bins, labels=labels, right=True, include_lowest=True
                    )
                    contagem_idade_mae = df_filtrado['Faixa Etária da Mãe'].value_counts(dropna=False).sort_index()
                    contagem_idade_mae = contagem_idade_mae.reindex(labels, fill_value=0)
                    total_idade_mae = contagem_idade_mae.sum()
            
                    if not contagem_idade_mae.empty and total_idade_mae > 0:
                        fig, ax = plt.subplots(figsize=(9, 7))
                        bars = ax.barh(contagem_idade_mae.index.astype(str), contagem_idade_mae.values, color='#92a9a7', edgecolor='#92a9a7')
                        for i, (categoria, valor) in enumerate(contagem_idade_mae.items()):
                            percentual = (valor / total_idade_mae) * 100
                            ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)
            
                        max_val_for_xlim = contagem_idade_mae.max()
                        if pd.notna(max_val_for_xlim) and max_val_for_xlim > 0:
                            ax.set_xlim(0, max_val_for_xlim * 1.2)
                        else:
                            ax.set_xlim(0, 10)
            
                        ax.set_title('Distribuição da Idade Gestacional da Mãe por Faixa Etária', fontsize=14)
                        ax.set_xlabel('Contagem de Mães', fontsize=12)
                        ax.set_ylabel('Faixa Etária da Mãe', fontsize=12)
                        ax.grid(axis='x', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.warning(f"Não há dados válidos para o gráfico de '{nome_coluna_idade_mae_original}'.")
                else:
                    st.warning(f"Coluna '{nome_coluna_idade_mae_original}' não encontrada ou DataFrame vazio.")

            with col2_saude_row2:
                st.subheader("Rede de Acompanhamento Pré-natal da Mãe")
                if 'Realizou o acompanhamento pré natal na rede:' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Realizou o acompanhamento pré natal na rede:'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#5fddaa', edgecolor='#5fddaa')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição da rede onde a mãe realizou pré natal')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Realizou o acompanhamento pré natal na rede:' não encontrada ou DataFrame vazio.")
            
            # Linha 3 de gráficos de Saúde
            col1_saude_row3, col2_saude_row3 = st.columns(2)
            
            with col1_saude_row3:
                st.subheader("Quando o Diagnóstico do Participante foi Realizado")
                if 'O diagnóstico do participante foi realizado no' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['O diagnóstico do participante foi realizado no'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#7696dc', edgecolor='#7696dc')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição de quando o diagnóstico do participante foi realizado')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Diagnóstico')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'O diagnóstico do participante foi realizado no' não encontrada ou DataFrame vazio.")

            with col2_saude_row3:
                st.subheader("Foi feito o cariótipo?")
                if 'Foi feito o cariótipo' in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['Foi feito o cariótipo'] = (
                        df_filtrado['Foi feito o cariótipo']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['Foi feito o cariótipo'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.bar(labels_formatados, contagem.values, color='#c5eea6', edgecolor='#c5eea6')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Distribuição dos participantes que fizeram o cariótipo')
                    ax.set_xlabel('Resposta')
                    ax.set_ylabel('Frequência')
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Foi feito o cariótipo' não encontrada ou DataFrame vazio.")

            # Linha 4 de gráficos de Saúde
            col1_saude_row4, col2_saude_row4 = st.columns(2)
            
            with col1_saude_row4:
                st.subheader("Irmãos do participante possuem alguma deficiência?")
                if 'Os irmãos do participante possui alguma deficiência' in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado['Os irmãos do participante possui alguma deficiência'] = (
                        df_filtrado['Os irmãos do participante possui alguma deficiência']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    contagem = df_filtrado['Os irmãos do participante possui alguma deficiência'].value_counts()
                    total = contagem.sum()

                    ordem_desejada = ['não', 'sim']
                    contagem = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem.index]

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.bar(labels_formatados, contagem.values, color='#ea664d', edgecolor='#ea664d')

                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        percentual = (height / total) * 100
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{percentual:.1f}%',
                                ha='center', va='bottom', fontsize=9)

                    ax.set_title('Distribuição de irmãos com alguma deficiência')
                    ax.set_xlabel('Resposta')
                    ax.set_ylabel('Frequência')
                    ax.set_xticklabels(labels_formatados, rotation=0, ha='center', fontsize=10)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Os irmãos do participante possui alguma deficiência' não encontrada ou DataFrame vazio.")

            with col2_saude_row4:
                st.subheader("Saúde Geral do Participante (Percepção do Cuidador)")
                nome_coluna_saude_geral = 'Você considera que a saúde do participante em geral é'
                if nome_coluna_saude_geral in df_filtrado.columns and not df_filtrado.empty:
                    df_filtrado[nome_coluna_saude_geral] = df_filtrado[nome_coluna_saude_geral].astype(str).str.lower().str.strip()
                    contagem_saude = df_filtrado[nome_coluna_saude_geral].value_counts(dropna=False)
                    ordem_saude_desejada = ['muito boa', 'boa', 'excelente', 'regular', 'ruim', 'muito ruim', 'nan']
                    contagem_saude = contagem_saude.reindex(ordem_saude_desejada, fill_value=0).dropna()
                    contagem_saude = contagem_saude[contagem_saude > 0]
                    total_saude = contagem_saude.sum()

                    if not contagem_saude.empty and total_saude > 0:
                        fig, ax = plt.subplots(figsize=(9, 7))
                        bars = ax.barh(contagem_saude.index.astype(str), contagem_saude.values, color='#ff9486', edgecolor='#ff9486')
                        for i, (categoria, valor) in enumerate(contagem_saude.items()):
                            if total_saude > 0:
                                percentual = (valor / total_saude) * 100
                                ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)
                            else:
                                ax.text(valor + 5, i, '0.0%', va='center', fontsize=9)

                        max_val_for_xlim = contagem_saude.max()
                        if pd.notna(max_val_for_xlim) and max_val_for_xlim > 0:
                            ax.set_xlim(0, max_val_for_xlim * 1.2)
                        else:
                            ax.set_xlim(0, 10)

                        ax.set_title('Distribuição de como o cuidador considera a saúde do participante em geral', fontsize=14)
                        ax.set_xlabel('Frequência', fontsize=12)
                        ax.set_ylabel('Saúde em geral', fontsize=12)
                        ax.grid(axis='x', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    else:
                        st.warning(f"Não há dados válidos para o gráfico de '{nome_coluna_saude_geral}'.")
                else:
                    st.warning(f"Coluna '{nome_coluna_saude_geral}' não encontrada ou DataFrame vazio.")

 
            # Linha 5 de gráficos de Saúde
            col1_saude_row5, col2_saude_row5 = st.columns(2)           

            with col1_saude_row5:
                st.subheader("Rede de Acompanhamento Psicológico")
                if 'Realiza/realizou acompanhamento psicológico na rede' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Realiza/realizou acompanhamento psicológico na rede'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#f5da7a', edgecolor='#f5da7a')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição onde o participante realiza/realizou acompanhamento psicológico')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Realiza/realizou acompanhamento psicológico na rede' não encontrada ou DataFrame vazio.")

            with col2_saude_row5:
                st.subheader("Rede de Estimulação Precoce")
            
                if 'Realizou o programa de intervenção/estimulação precoce na rede' in df_filtrado.columns and not df_filtrado.empty:
                    rede_estimulacao = df_filtrado['Realizou o programa de intervenção/estimulação precoce na rede'].copy()
                    rede_estimulacao.replace('Particular', 'Privada', inplace=True)

                    contagem = rede_estimulacao.value_counts()
                    total = contagem.sum()
                    porcentagem = (contagem / total) * 100
            
                    fig, ax = plt.subplots(figsize=(6, 5))
            
                    cor_unica = sns.color_palette("viridis", 1)[0]
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color=cor_unica, edgecolor=cor_unica)
            
                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = porcentagem.loc[categoria]
                        ax.text(valor + (contagem.max() * 0.05),
                                i,
                                f'{percentual:.1f}%',
                                va='center',
                                fontsize=9,
                                color='black')
            
                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.invert_yaxis()
                    ax.set_title('Distribuição onde o participante realizou estimulação precoce')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
            
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
            
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                else:
                    st.warning("Coluna 'Realizou o programa de intervenção/estimulação precoce na rede' não encontrada ou DataFrame vazio.")

            # Linha 6 de gráficos de Saúde
            col1_saude_row6, col2_saude_row6 = st.columns(2)
            
            with col1_saude_row6:
                st.subheader("Rede de Acompanhamento com Clínico Geral")
                # Atenção: 'Se sim, o acompanhamento ocorre na rede1:' é o nome real da coluna
                if 'Se sim, o acompanhamento ocorre na rede1:' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Se sim, o acompanhamento ocorre na rede1:'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#a35572', edgecolor='#a35572')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição onde o participante faz acompanhamento com clínico geral')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Se sim, o acompanhamento ocorre na rede1:' não encontrada ou DataFrame vazio.")

            with col2_saude_row6:
                st.subheader("Rede de Acompanhamento com Dentista")
                # Atenção: 'Se sim, o acompanhamento ocorre na rede2:' é o nome real da coluna
                if 'Se sim, o acompanhamento ocorre na rede2:' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Se sim, o acompanhamento ocorre na rede2:'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#8ccc81', edgecolor='#8ccc81')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição onde o participante faz acompanhamento com dentista')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Se sim, o acompanhamento ocorre na rede2:' não encontrada ou DataFrame vazio.")

            # Linha 7 de gráficos de Saúde
            col1_saude_row7, col2_saude_row7 = st.columns(2)            

            with col1_saude_row7:
                st.subheader("Rede de Acompanhamento com Nutricionista")
                # Atenção: 'Se sim, o acompanhamento ocorre na rede3:' é o nome real da coluna
                if 'Se sim, o acompanhamento ocorre na rede3:' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Se sim, o acompanhamento ocorre na rede3:'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#d698b1', edgecolor='#d698b1')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição onde o participante faz acompanhamento com nutricionista')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Se sim, o acompanhamento ocorre na rede3:' não encontrada ou DataFrame vazio.")


            with col2_saude_row7:
                st.subheader("Rede de Acompanhamento com Oftalmologista")
                # Atenção: 'Se sim, o acompanhamento ocorre na rede4:' é o nome real da coluna
                if 'Se sim, o acompanhamento ocorre na rede4:' in df_filtrado.columns and not df_filtrado.empty:
                    contagem = df_filtrado['Se sim, o acompanhamento ocorre na rede4:'].value_counts()
                    total = contagem.sum()

                    fig, ax = plt.subplots(figsize=(6, 5))
                    bars = ax.barh(contagem.index.astype(str), contagem.values, color='#5b7c8d', edgecolor='#5b7c8d')

                    for i, (categoria, valor) in enumerate(contagem.items()):
                        percentual = (valor / total) * 100
                        ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)

                    ax.set_xlim(0, contagem.max() * 1.2)
                    ax.set_title('Distribuição onde o participante faz acompanhamento com oftalmologista')
                    ax.set_xlabel('Frequência')
                    ax.set_ylabel('Rede')
                    ax.grid(axis='x', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Coluna 'Se sim, o acompanhamento ocorre na rede4:' não encontrada ou DataFrame vazio.")

            # Linha 8 de gráficos de Saúde
            col1_saude_row8, col2_saude_row8 = st.columns(2) 
            
            with col1_saude_row8:

                # --- Gráfico de Diagnósticos Psicológicos (sem "Sem Diagnóstico") ---
                st.subheader("Distribuição de Diagnósticos Psicológicos")
                
                # 1. Definir colunas de diagnóstico (removendo 'Não possui')
                diagnosticos_cols = [
                    'O participante possui diagnóstico médico de (Ansiedade)',
                    'O participante possui diagnóstico médico de (Depressão)',
                    'O participante possui diagnóstico médico de (Outros distúrbios de humor)'
                ]
                
                # Filtrar colunas existentes
                cols_existentes = [col for col in diagnosticos_cols if col in df_filtrado.columns]
                
                if cols_existentes and not df_filtrado.empty:
                    # 2. Processar os dados
                    df_diag = df_filtrado[cols_existentes].copy()
                    
                    # Converter para valores booleanos
                    for col in cols_existentes:
                        df_diag[col] = df_diag[col].astype(str).str.strip().str.lower()
                        df_diag[col] = df_diag[col].map({'sim': True, 'não': False, 'true': True, 'false': False}).fillna(False)
                    
                    # Contar os diagnósticos
                    contagem = df_diag.sum()
                    total = len(df_diag)
                    porcentagem = (contagem / total) * 100
                    
                    # Renomear para nomes mais amigáveis
                    nomes_amigaveis = {
                        'O participante possui diagnóstico médico de (Ansiedade)': 'Ansiedade',
                        'O participante possui diagnóstico médico de (Depressão)': 'Depressão',
                        'O participante possui diagnóstico médico de (Outros distúrbios de humor)': 'Outros Distúrbios'
                    }
                    contagem.index = contagem.index.map(nomes_amigaveis)
                    porcentagem.index = porcentagem.index.map(nomes_amigaveis)
                    
                    # 3. Criar o gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Usar palette do Seaborn para cores consistentes
                    cores = sns.color_palette("viridis", 1)
                    barras = ax.bar(contagem.index, contagem.values, color=cores)
                    
                    # Adicionar valores e porcentagens
                    for barra, valor, pct in zip(barras, contagem.values, porcentagem.values):
                        altura = barra.get_height()
                        ax.text(barra.get_x() + barra.get_width()/2, altura + 0.5,
                               f'{valor} ({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
                    
                    # Configurações do gráfico
                    ax.set_title('Prevalência de Diagnósticos Psicológicos', fontsize=14, pad=20)
                    ax.set_xlabel('Tipo de Diagnóstico', fontsize=12)
                    ax.set_ylabel('Frquência', fontsize=12)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    ax.set_ylim(0, contagem.max() * 1.2)
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                        
                else:
                    st.warning("Nenhum dado de diagnóstico psicológico disponível após os filtros aplicados.")

            # --- Seção: Distribuição de Imunizações (Vacinas) ---
            with col2_saude_row8:
                st.subheader("Distribuição de Imunizações (Vacinas)")
                colunas_imunizacao = [
                    'Recebeu alguma das imunizações a seguir (Palivizumabe)',
                    'Recebeu alguma das imunizações a seguir (BCG)',
                    'Recebeu alguma das imunizações a seguir (Hepatite B)',
                    'Recebeu alguma das imunizações a seguir (Rotavirus)',
                    'Recebeu alguma das imunizações a seguir (Tríplice bacteriana)',
                    'Recebeu alguma das imunizações a seguir (Haemophilus influenzae b)',
                    'Recebeu alguma das imunizações a seguir (Poliomielite vírus inativado)',
                    'Recebeu alguma das imunizações a seguir (Poliomielite oral)',
                    'Recebeu alguma das imunizações a seguir (Pneumocócicas conjugadas)',
                    'Recebeu alguma das imunizações a seguir (Meningocócicas conjugadas)',
                    'Recebeu alguma das imunizações a seguir (Meningocócica B)',
                    'Recebeu alguma das imunizações a seguir (Influenza)',
                    'Recebeu alguma das imunizações a seguir (Febre amarela)',
                    'Recebeu alguma das imunizações a seguir (Tríplice viral)',
                    'Recebeu alguma das imunizações a seguir (Varicela)',
                    'Recebeu alguma das imunizações a seguir (Hepatite A)',
                    'Recebeu alguma das imunizações a seguir (HPV)',
                    'Recebeu alguma das imunizações a seguir (Difteria e tétano adulto)',
                    'Recebeu alguma das imunizações a seguir (Herpes zoster)',
                    'Recebeu alguma das imunizações a seguir (Dengue)',
                    'Recebeu alguma das imunizações a seguir (Recebeu todas)'
                ]
            
                colunas_existentes_imunizacao = [col for col in colunas_imunizacao if col in df_filtrado.columns]
            
                if not colunas_existentes_imunizacao:
                    st.warning("Nenhuma coluna de imunização (vacinas) foi encontrada no DataFrame. Verifique os nomes.")
                else:
                    contagens_vacinas = {}
                    total_participantes_para_porcentagem = len(df_filtrado)
            
                    if total_participantes_para_porcentagem == 0:
                        st.info("Nenhum participante encontrado com os filtros selecionados para exibir imunizações.")
                    else:
                        df_temp_imunizacao = df_filtrado.copy()
            
                        for col_imunizacao_single in colunas_existentes_imunizacao:
                            df_temp_imunizacao[col_imunizacao_single] = df_temp_imunizacao[col_imunizacao_single].astype(str).str.strip().str.lower()
                            df_temp_imunizacao[col_imunizacao_single] = df_temp_imunizacao[col_imunizacao_single].apply(lambda x: 'Sim' if x == 'sim' else 'Não')
                        
                            count_sim = df_temp_imunizacao[df_temp_imunizacao[col_imunizacao_single] == 'Sim'].shape[0]
                        
                            if '(' in col_imunizacao_single and ')' in col_imunizacao_single:
                                nome_vacina_limpo = col_imunizacao_single.split('(')[-1].replace(')', '')
                            else:
                                nome_vacina_limpo = col_imunizacao_single
            
                            if count_sim > 0:
                                contagens_vacinas[nome_vacina_limpo] = count_sim
            
                        if not contagens_vacinas:
                            st.info("Nenhuma vacina possui contagens positivas para plotar com os filtros selecionados.")
                        else:
                            contagem_series = pd.Series(contagens_vacinas).sort_values(ascending=True)
            
                            fig, ax = plt.subplots(figsize=(8, max(6, len(contagem_series) * 0.4)))
            
                            cor_unica = '#f5da7a'
                            bars = ax.barh(contagem_series.index, contagem_series.values, color=cor_unica, edgecolor=cor_unica)
            
                            for i, (vacina, valor) in enumerate(contagem_series.items()):
                                percentual = (valor / total_participantes_para_porcentagem) * 100
                                ax.text(valor + (contagem_series.max() * 0.02),
                                        i,
                                        f'{percentual:.1f}%',
                                        va='center',
                                        fontsize=9,
                                        color='black')
            
                            ax.set_xlim(0, contagem_series.max() * 1.25)
                            ax.set_title('Distribuição de imunizações/vacinas entre os participantes', fontsize=16)
                            ax.set_xlabel('Frequência')
                            ax.set_ylabel('Imunizante/vacina')
                            ax.grid(axis='x', linestyle='--', alpha=0.5)
                            plt.tight_layout()
            
                            for spine in ax.spines.values():
                                spine.set_visible(True)
                                spine.set_linewidth(1.2)
            
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

            # ------------------------------------------------------------------------
                   
            # --- Prevalência de Morbidades e/ou Doenças ---
            st.subheader("Prevalência de Morbidades e/ou Doenças")
            col_morbidades = [col for col in df_filtrado.columns if 'Apresenta algumas das morbidades' in col]
            
            if col_morbidades and not df_filtrado.empty:
                df_morbidades_temp = df_filtrado[col_morbidades].copy()
                for col in col_morbidades:
                    df_morbidades_temp[col] = df_morbidades_temp[col].astype(str).str.strip().str.lower()
                    df_morbidades_temp[col] = df_morbidades_temp[col].map({'sim': 'Sim', 'não': 'Não', 'nan': 'Não'}).fillna('Não')
            
                df_morbidades_long = df_morbidades_temp.melt(var_name='Morbidade', value_name='Status')
                df_morbidades_sim = df_morbidades_long[df_morbidades_long['Status'] == 'Sim']
                contagem_morbidades = df_morbidades_sim['Morbidade'].value_counts().reset_index()
                contagem_morbidades.columns = ['Morbidade', 'Contagem']
                contagem_morbidades['Morbidade Limpa'] = contagem_morbidades['Morbidade'].str.replace('Apresenta algumas das morbidades (', '').str.replace(')', '')
            
                if not contagem_morbidades.empty:
                    total_morbidades_sim = contagem_morbidades['Contagem'].sum()
                    contagem_morbidades['Porcentagem'] = (contagem_morbidades['Contagem'] / total_morbidades_sim) * 100
                    contagem_morbidades['Morbidade_ID'] = range(1, len(contagem_morbidades) + 1)
            
                    # Criar figura com dois subplots - um para a legenda e outro para o treemap
                    fig = plt.figure(figsize=(14, 12))
                    
                    # Subplot 1 - apenas para a legenda (topo)
                    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=2)
                    ax1.axis('off')
                    
                    # Subplot 2 - para o treemap (parte inferior)
                    ax2 = plt.subplot2grid((10, 1), (2, 0), rowspan=8)
                    
                    colors = [plt.cm.viridis(i / float(len(contagem_morbidades['Contagem']))) for i in range(len(contagem_morbidades['Contagem']))]
            
                    # Gerar a legenda no subplot superior
                    legend_labels = []
                    for index, row in contagem_morbidades.iterrows():
                        legend_labels.append(
                            f"{row['Morbidade_ID']}: {row['Morbidade Limpa']} "
                            f"({row['Contagem']} - {row['Porcentagem']:.1f}%)"
                        )
            
                    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], ec="none") for i in range(len(colors))]
                    
                    # Criar legenda no subplot superior
                    ax1.legend(legend_patches, legend_labels, title="Detalhes das Morbidades",
                              loc='center', ncol=2, fontsize=9)
                    
                    # Gerar o treemap no subplot inferior
                    squarify.plot(
                        sizes=contagem_morbidades['Contagem'],
                        label=contagem_morbidades['Morbidade_ID'].astype(str),
                        color=colors,
                        alpha=.8,
                        pad=True,
                        text_kwargs={'fontsize': 10, 'weight': 'bold', 'color': 'white'},
                        ax=ax2
                    )
            
                    ax2.set_title('Distribuição das Morbidades', fontsize=12)
                    ax2.axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.warning("Não há dados de morbidades 'Sim' para gerar a treemap após os filtros.")
            else:
                st.warning("Não foram encontradas colunas de morbidades ou DataFrame vazio.")

            # -----------------------------------------
            col1_saude_row8, col2_saude_row8 = st.columns(2) 
    
            with col1_saude_row8:
                # --- Gráfico de Doenças Apesar das Imunizações ---
                st.subheader("Doenças Apesar das Imunizações")
                
                # 1. Definir as colunas de imunização (Vacinas)
                colunas_imunizacao = [
                    'Apesar das imunizações, teve alguma dessas doenças (Tuberculose)',
                    'Apesar das imunizações, teve alguma dessas doenças (Hepatite B)',
                    'Apesar das imunizações, teve alguma dessas doenças (Poliomielite)',
                    'Apesar das imunizações, teve alguma dessas doenças (Diarreia por rotavírus)',
                    'Apesar das imunizações, teve alguma dessas doenças (Dfiteria)',
                    'Apesar das imunizações, teve alguma dessas doenças (Tétano)',
                    'Apesar das imunizações, teve alguma dessas doenças (Coqueluche)',
                    'Apesar das imunizações, teve alguma dessas doenças (Haemophilus influenzae B)',
                    'Apesar das imunizações, teve alguma dessas doenças (Pneumonia)',
                    'Apesar das imunizações, teve alguma dessas doenças (Meningite)',
                    'Apesar das imunizações, teve alguma dessas doenças (Febre amarela)',
                    'Apesar das imunizações, teve alguma dessas doenças (Sarampo)',
                    'Apesar das imunizações, teve alguma dessas doenças (Caxumba)',
                    'Apesar das imunizações, teve alguma dessas doenças (Rubéola)',
                    'Apesar das imunizações, teve alguma dessas doenças (Varicela/catapora)',
                    'Apesar das imunizações, teve alguma dessas doenças (Hepatite A)',
                    'Apesar das imunizações, teve alguma dessas doenças (Papiloma vírus)',
                    'Apesar das imunizações, teve alguma dessas doenças (Dengue)',
                    'Apesar das imunizações, teve alguma dessas doenças (Não teve)',
                ]
                
                # Filtrar apenas as colunas que existem no DataFrame filtrado
                colunas_existentes_imunizacao = [col for col in colunas_imunizacao if col in df_filtrado.columns]
                
                if not colunas_existentes_imunizacao:
                    st.warning("Nenhuma coluna de doenças pós-imunização encontrada após os filtros aplicados.")
                else:
                    contagens_vacinas = {}
                    total_participantes = len(df_filtrado)
                    
                    if total_participantes == 0:
                        st.info("Nenhum participante encontrado com os filtros selecionados.")
                    else:
                        df_temp = df_filtrado.copy()
                        
                        for col_imunizacao in colunas_existentes_imunizacao:
                            # Padronizar valores
                            df_temp[col_imunizacao] = df_temp[col_imunizacao].astype(str).str.strip().str.lower()
                            df_temp[col_imunizacao] = df_temp[col_imunizacao].apply(lambda x: 'Sim' if x == 'sim' else 'Não')
                            
                            # Contar ocorrências
                            count_sim = df_temp[df_temp[col_imunizacao] == 'Sim'].shape[0]
                            
                            # Extrair nome limpo
                            if '(' in col_imunizacao and ')' in col_imunizacao:
                                nome_doenca = col_imunizacao.split('(')[-1].replace(')', '')
                            else:
                                nome_doenca = col_imunizacao
                            
                            if count_sim > 0:
                                contagens_vacinas[nome_doenca] = count_sim
                        
                        if not contagens_vacinas:
                            st.info("Nenhuma doença reportada após imunização com os filtros selecionados.")
                        else:
                            contagem_series = pd.Series(contagens_vacinas).sort_values(ascending=True)
                            
                            fig, ax = plt.subplots(figsize=(8, max(6, len(contagem_series) * 0.4)))
                            bars = ax.barh(contagem_series.index, contagem_series.values, color='#79aba2', edgecolor='#79aba2')
                            
                            # Adicionar porcentagens
                            for i, (doenca, valor) in enumerate(contagem_series.items()):
                                percentual = (valor / total_participantes) * 100
                                ax.text(valor + (contagem_series.max() * 0.02), i, 
                                        f'{percentual:.1f}%', va='center', fontsize=9)
                            
                            ax.set_xlim(0, contagem_series.max() * 1.25)
                            ax.set_title('Doenças contraídas apesar das imunizações', fontsize=14)
                            ax.set_xlabel('Frequência')
                            ax.set_ylabel('Doenças')
                            ax.grid(axis='x', linestyle='--', alpha=0.5)
                            
                            for spine in ax.spines.values():
                                spine.set_visible(True)
                                spine.set_linewidth(1.2)
                            
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
            
            with col2_saude_row8:
                # --- Gráfico de Gravidade da COVID ---
                st.subheader("Gravidade da COVID")
                
                if 'Gravidade da COVID do participante' not in df_filtrado.columns:
                    st.warning("Coluna 'Gravidade da COVID do participante' não encontrada.")
                else:
                    contagem = df_filtrado['Gravidade da COVID do participante'].value_counts()
                    total = contagem.sum()
                    
                    if total == 0:
                        st.info("Nenhum dado disponível sobre gravidade da COVID com os filtros aplicados.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.barh(contagem.index.astype(str), contagem.values, color='#c3c2ff', edgecolor='#c3c2ff')
                        
                        # Adicionar porcentagens
                        for i, (categoria, valor) in enumerate(contagem.items()):
                            percentual = (valor / total) * 100
                            ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)
                        
                        ax.set_xlim(0, contagem.max() * 1.2)
                        ax.set_title('Gravidade da COVID entre participantes', fontsize=14)
                        ax.set_xlabel('Frequência')
                        ax.set_ylabel('Gravidade')
                        ax.grid(axis='x', linestyle='--', alpha=0.5)
                        
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
    
            # ---------------------
            # Tabela de distribuição de doses de vacina COVID-19
            st.subheader("Distribuição de Doses de Vacina COVID-19")
            
            # Definir as colunas de interesse
            doses = [
                'Tomou a 1ª dose da COVID',
                'Tomou a 2ª dose da COVID',
                'Tomou a 3ª dose da COVID',
                'Tomou a 4ª dose da COVID',
                'Tomou a 5ª dose da COVID',
                'Tomou a 6ª dose da COVID'
            ]
            
            # Verificar quais colunas existem no DataFrame
            doses_existentes = [col for col in doses if col in df_filtrado.columns]
            
            if not doses_existentes:
                st.warning("Nenhuma informação sobre doses de vacina disponível.")
            else:
                # Criar DataFrame vazio com o formato desejado
                tabela_resultado = pd.DataFrame(index=['Não', 'Sim'], columns=doses_existentes)
                
                # Preencher o DataFrame com as contagens
                for dose in doses_existentes:
                    # Contar valores 'Sim' e 'Não', considerando NaN como 'Não'
                    contagens = df_filtrado[dose].fillna('Não').value_counts()
                    
                    # Preencher os valores na tabela
                    tabela_resultado.loc['Não', dose] = contagens.get('Não', 0)
                    tabela_resultado.loc['Sim', dose] = contagens.get('Sim', 0)
                
                # Converter todos valores para inteiros
                tabela_resultado = tabela_resultado.fillna(0).astype(int)
                
                # Exibir a tabela formatada
                st.dataframe(
                    tabela_resultado.style
                        .format("{:d}")  # Formato inteiro sem decimais
                        .set_properties(**{'text-align': 'center'})
                        .set_table_styles([{
                            'selector': 'th',
                            'props': [('background-color', '#f0f2f6'), 
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center')]
                        }]),
                    use_container_width=True
                )
                
                # Adicionar informação do total de respondentes
                st.caption(f"Total de participantes analisados: {len(df_filtrado)}")
            # ---------------------
            # --- GRÁFICO DE MEDICAÇÕES (FULL WIDTH) ---
            st.subheader("Medicações Utilizadas pelos Participantes")
            
            colunas_medicacoes = [
                'O participante toma alguma dessas medicações (Antidepressivo/ansiolitico)',
                'O participante toma alguma dessas medicações (Anti-hipertensivo/ICC)',
                'O participante toma alguma dessas medicações (Antidiabético em geral)',
                'O participante toma alguma dessas medicações (Estatina/antidislipidemico)',
                'O participante toma alguma dessas medicações (Repositor hormonal da tireoide)',
                'O participante toma alguma dessas medicações (Vitamina D)',
                'O participante toma alguma dessas medicações (Glicocorticoide)',
                'O participante toma alguma dessas medicações (Outros)',
                'O participante toma alguma dessas medicações (Não toma medicação)',
            ]
            
            colunas_existentes = [col for col in colunas_medicacoes if col in df_filtrado.columns]
            
            if not colunas_existentes:
                st.warning("Nenhuma informação sobre medicações disponível.")
            else:
                contagens = {}
                total = len(df_filtrado)
                
                for col in colunas_existentes:
                    # Padronizar valores
                    df_filtrado[col] = df_filtrado[col].astype(str).str.strip().str.lower()
                    count_sim = (df_filtrado[col] == 'sim').sum()
                    
                    # Extrair nome da medicação
                    nome = col.split('(')[-1].replace(')', '') if '(' in col else col
                    contagens[nome] = count_sim
                
                if not contagens:
                    st.info("Nenhuma medicação com contagens positivas.")
                else:
                    serie_contagens = pd.Series(contagens).sort_values()
                    
                    fig, ax = plt.subplots(figsize=(10, max(4, len(serie_contagens)*0.5)))  # Ajustado para full width
                    bars = ax.barh(serie_contagens.index, serie_contagens.values, color='#fbc599')
                    
                    # Adicionar porcentagens
                    for i, (med, valor) in enumerate(serie_contagens.items()):
                        percent = (valor/total)*100
                        ax.text(valor + serie_contagens.max()*0.02, i, 
                                f'({percent:.1f}%)', va='center', fontsize=10)
                    
                    ax.set_xlim(0, serie_contagens.max()*1.3)  # Aumentado espaço para texto
                    ax.set_title('Uso de Medicações pelos Participantes', fontsize=16, pad=20)
                    ax.set_xlabel('Frequência', fontsize=12)
                    ax.set_ylabel('Tipo de Medicação', fontsize=12)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    # Melhorar a estética
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(1.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            col1_saude_row9, col2_saude_row9 = st.columns(2) 
            
            with col1_saude_row9:
                # --- GRÁFICO DE ATIVIDADE FÍSICA ---
                st.subheader("Prática de Atividade Física")
                
                if 'Pratica alguma atividade física ou esporte' not in df_filtrado.columns:
                    st.warning("Dados sobre atividade física não disponíveis.")
                else:
                    # Processar dados
                    df_filtrado['Atividade_Fisica'] = (
                        df_filtrado['Pratica alguma atividade física ou esporte']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    
                    contagem = df_filtrado['Atividade_Fisica'].value_counts()
                    total = contagem.sum()
                    
                    # Ordenar e formatar
                    ordem = ['não', 'sim']
                    contagem = contagem.reindex(ordem).dropna()
                    labels = [s.capitalize() for s in contagem.index]
                    
                    if contagem.empty:
                        st.info("Nenhum dado disponível sobre atividade física.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels, contagem.values, color='#aef055')
                        
                        # Adicionar porcentagens
                        for bar in bars:
                            height = bar.get_height()
                            percent = (height/total)*100
                            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                                    f'{percent:.1f}%', ha='center', va='bottom')
                        
                        ax.set_title('Prática de Atividade Física', fontsize=14)
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Número de Participantes')
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        
            # ---------------------            
            with col2_saude_row9:
                # --- GRÁFICO DE FREQUÊNCIA DE ATIVIDADE FÍSICA ---
                st.subheader("Frequência Semanal de Atividade Física")
                
                if 'O participante pratica atividade física ou esporte quantas vezes na semana' not in df_filtrado.columns:
                    st.warning("Dados sobre frequência de atividade física não disponíveis.")
                else:
                    # Processar os dados
                    contagem = df_filtrado['O participante pratica atividade física ou esporte quantas vezes na semana'].value_counts()
                    total = contagem.sum()
                    
                    if total == 0:
                        st.info("Nenhum dado disponível sobre frequência de atividade física.")
                    else:
                        # Criar gráfico
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.barh(contagem.index.astype(str), contagem.values, color='#9b5f7b', edgecolor='#9b5f7b')
                        
                        # Adicionar porcentagens
                        for i, (categoria, valor) in enumerate(contagem.items()):
                            percentual = (valor / total) * 100
                            ax.text(valor + 5, i, f'{percentual:.1f}%', va='center', fontsize=9)
                        
                        # Configurações do gráfico
                        ax.set_xlim(0, contagem.max() * 1.2)
                        ax.set_title('Frequência Semanal de Atividade Física', fontsize=14)
                        ax.set_xlabel('Número de Participantes')
                        ax.set_ylabel('Vezes por Semana')
                        ax.grid(axis='x', linestyle='--', alpha=0.5)
                        
                        # Exibir no Streamlit
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            # Criar layout de duas colunas
            col1_saude_row10, col2_saude_row10 = st.columns(2) 
            
            with col1_saude_row10:
                # --- GRÁFICO DE HÁBITOS ALIMENTARES ---
                st.subheader("Hábitos Alimentares Saudáveis")
                
                if 'Você considera que o participante possui hábitos alimentares saudáveis' not in df_filtrado.columns:
                    st.warning("Dados sobre hábitos alimentares não disponíveis.")
                else:
                    # Processar os dados
                    df_filtrado['Habitos_Alimentares'] = (
                        df_filtrado['Você considera que o participante possui hábitos alimentares saudáveis']
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    
                    contagem = df_filtrado['Habitos_Alimentares'].value_counts()
                    total = contagem.sum()
                    
                    if total == 0:
                        st.info("Nenhum dado disponível sobre hábitos alimentares.")
                    else:
                        # Ordenar e formatar
                        ordem_desejada = ['não', 'sim']
                        contagem = contagem.reindex(ordem_desejada).dropna()
                        labels = [s.capitalize() for s in contagem.index]
                        
                        # Criar gráfico
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels, contagem.values, color='#b7833a', edgecolor='#b7833a')
                        
                        # Adicionar porcentagens
                        for bar in bars:
                            height = bar.get_height()
                            percentual = (height / total) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, 
                                    f'{percentual:.1f}%', ha='center', va='bottom', fontsize=9)
                        
                        # Configurações do gráfico
                        ax.set_title('Hábitos Alimentares Saudáveis', fontsize=14)
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Número de Participantes')
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        
                        # Exibir no Streamlit
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
            # ---------------------
            with col2_saude_row10:
                # --- GRÁFICO DE ATENDIMENTO DE SAÚDE PÚBLICO ---
                st.subheader("Atendimento de Saúde Público Adequado")
                
                coluna_saude = 'Você considera que o participante possui atendimento de saúde adequado pelo sistema público para atender suas necessidades'
                
                if coluna_saude not in df_filtrado.columns:
                    st.warning("Dados sobre atendimento de saúde público não disponíveis.")
                else:
                    # Processar os dados
                    df_filtrado['Atendimento_Saude'] = (
                        df_filtrado[coluna_saude]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                    )
                    
                    contagem = df_filtrado['Atendimento_Saude'].value_counts()
                    total = contagem.sum()
                    
                    if total == 0:
                        st.info("Nenhum dado disponível sobre atendimento de saúde público.")
                    else:
                        # Ordenar e formatar
                        ordem_desejada = ['não', 'sim']
                        contagem = contagem.reindex(ordem_desejada).dropna()
                        labels = [s.capitalize() for s in contagem.index]
                        
                        # Criar gráfico
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels, contagem.values, color='#838689', edgecolor='#838689')
                        
                        # Adicionar porcentagens
                        for bar in bars:
                            height = bar.get_height()
                            percentual = (height / total) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, 
                                   f'{percentual:.1f}%', ha='center', va='bottom', fontsize=9)
                        
                        # Configurações do gráfico
                        ax.set_title('Atendimento de Saúde Público Adequado', fontsize=14)
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Número de Participantes')
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        
                        # Exibir no Streamlit
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    
            # -----

        with tab6:
            st.header("Características de Participação, Autonomia e Interação Social")
            # --- ABA 6: Características de Pariticipação, Autonomia e Interação Social (Novos Gráficos Matplotlib em Colunas) ---
            col_autonomia, col_deslocamento = st.columns(2)
            col_necessidades, col_relacionamento = st.columns(2)
            col_interacao_social, col_respeito = st.columns(2)
            
            # --- Gráfico 1: Autonomia para Tomar Decisões ---
            with col_autonomia:
                st.subheader("Autonomia para Tomar Decisões")
                coluna_autonomia = 'Você considera que o participante possui autonomia para tomar decisões e tem a chance de escolher as coisas que deseja (escolher as coisas para sua satisfação pessoal)?'

                if coluna_autonomia in df_filtrado.columns and not df_filtrado.empty:
                    temp_series = df_filtrado[coluna_autonomia].astype(str).str.lower().str.strip()
                    contagem = temp_series.value_counts()
                    total_participantes = len(df_filtrado)
        
                    # Ordenar para garantir "Não" antes de "Sim" se ambos existirem
                    ordem_desejada = ['não', 'sim']
                    contagem_ordenada = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem_ordenada.index]
        
                    if total_participantes == 0 or contagem_ordenada.empty:
                        st.info("Nenhum dado para exibir para Autonomia para Tomar Decisões com os filtros selecionados.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels_formatados, contagem_ordenada.values, color='#8ccc81', edgecolor='#8ccc81')
        
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentual = (height / total_participantes) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                    f'{percentual:.1f}%', # Apenas porcentagem
                                    ha='center', va='bottom', fontsize=9, color='black')
        
                        ax.set_ylim(0, contagem_ordenada.max() * 1.15)
                        ax.set_title('Possui autonomia para tomar decisões') # Título mais conciso
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Frequência')
                        ax.tick_params(axis='x', rotation=0, labelsize=10) # Ajuste para Streamlit
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning(f"Coluna '{coluna_autonomia}' não encontrada ou DataFrame vazio para esta seleção.")

    # --- Gráfico 2: Deslocamento Independentemente pela Cidade ---
            with col_deslocamento:
                st.subheader("Deslocamento Independente")
                coluna_deslocamento = 'O participante se desloca independentemente pela cidade'

                if coluna_deslocamento in df_filtrado.columns and not df_filtrado.empty:
                    temp_series = df_filtrado[coluna_deslocamento].astype(str).str.lower().str.strip()
                    contagem = temp_series.value_counts()
                    total_participantes = len(df_filtrado)
        
                    ordem_desejada = ['não', 'sim']
                    contagem_ordenada = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem_ordenada.index]
        
                    if total_participantes == 0 or contagem_ordenada.empty:
                        st.info("Nenhum dado para exibir para Deslocamento Independente com os filtros selecionados.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels_formatados, contagem_ordenada.values, color='#bcc499', edgecolor='#bcc499')
        
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentual = (height / total_participantes) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                    f'{percentual:.1f}%', # Apenas porcentagem
                                    ha='center', va='bottom', fontsize=9, color='black')
        
                        ax.set_ylim(0, contagem_ordenada.max() * 1.15)
                        ax.set_title('Se desloca independentemente pela cidade') # Título mais conciso
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Frequência')
                        ax.tick_params(axis='x', rotation=0, labelsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning(f"Coluna '{coluna_deslocamento}' não encontrada ou DataFrame vazio para esta seleção.")

    # --- GRÁFICO 3: Necessidades Pessoais ---
            with col_necessidades: # Usando a primeira coluna da segunda linha
                st.subheader("Realização de Necessidades Pessoais")
                coluna_necessidades = 'Na maioria das vezes, você considera o participante na realização de suas necessidades pessoais como'
    
                if coluna_necessidades in df_filtrado.columns and not df_filtrado.empty:
                    temp_series = df_filtrado[coluna_necessidades].astype(str).str.lower().str.strip()
                    contagem = temp_series.value_counts(dropna=False)
    
                    if 'nan' in contagem.index:
                        contagem = contagem.rename(index={'nan': 'Não Preenchido'})
                        # Se você NÃO QUISER que 'Não Preenchido' apareça no gráfico, descomente a linha abaixo:
                        # contagem = contagem.drop('Não Preenchido', errors='ignore')
    
                    total_participantes = len(df_filtrado) # Usar df_filtrado para o total de participantes
    
                    contagem = contagem.sort_values(ascending=True)
    
                    if total_participantes == 0 or contagem.empty:
                        st.info("Nenhum dado para exibir para Necessidades Pessoais com os filtros selecionados.")
                    else:
                        fig, ax = plt.subplots(figsize=(10, max(6, len(contagem) * 0.7))) # Ajuste o figsize para barras horizontais
                        labels_formatados = [s.capitalize() for s in contagem.index]
                        bars = ax.barh(labels_formatados, contagem.values, color='#f8afb8', edgecolor='#f8afb8')
    
                        for i, (categoria_original, valor) in enumerate(contagem.items()):
                            percentual = (valor / total_participantes) * 100
                            ax.text(valor + (contagem.max() * 0.02), i, f'{percentual:.1f}%', va='center', fontsize=9)
    
                        ax.set_xlim(0, contagem.max() * 1.15)
                        ax.set_title('Como o cuidador considera o participante na realização de necessidades pessoais')
                        ax.set_xlabel('Frequência')
                        ax.set_ylabel('Forma de Independência')
                        ax.grid(axis='x', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning(f"Coluna '{coluna_necessidades}' não encontrada ou DataFrame vazio para esta seleção.")                

    # --- Gráfico 4: Participante se Relaciona com Diferentes Pessoas ---
            with col_relacionamento:
                st.subheader("Relacionamento Interpessoal")
                coluna_relacionamento = 'Você considera que o participante se relaciona com diferentes pessoas, tem amigos e se dá bem com as pessoas'
    
                if coluna_relacionamento in df_filtrado.columns and not df_filtrado.empty:
                    temp_series = df_filtrado[coluna_relacionamento].astype(str).str.lower().str.strip()
                    contagem = temp_series.value_counts()
                    total_participantes = len(df_filtrado)
        
                    ordem_desejada = ['não', 'sim']
                    contagem_ordenada = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem_ordenada.index]
        
                    if total_participantes == 0 or contagem_ordenada.empty:
                        st.info("Nenhum dado para exibir para Relacionamento Interpessoal com os filtros selecionados.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels_formatados, contagem_ordenada.values, color='#525574', edgecolor='#525574')
        
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentual = (height / total_participantes) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                    f'{percentual:.1f}%', # Apenas porcentagem
                                    ha='center', va='bottom', fontsize=9, color='black')
        
                        ax.set_ylim(0, contagem_ordenada.max() * 1.15)
                        ax.set_title('Se relaciona com diferentes pessoas e tem amigos') # Título mais conciso
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Frequência')
                        ax.tick_params(axis='x', rotation=0, labelsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning(f"Coluna '{coluna_relacionamento}' não encontrada ou DataFrame vazio para esta seleção.")

    # --- Gráfico 5: Interação Social (frequenta lugares) ---
            with col_interacao_social:
                st.subheader("Interação Social Voluntária")
                coluna_interacao_social = 'Você considera que o participante, por vontade própria interage socialmente, frequentando diferentes lugares da cidade ou do bairro'

                if coluna_interacao_social in df_filtrado.columns and not df_filtrado.empty:
                    temp_series = df_filtrado[coluna_interacao_social].astype(str).str.lower().str.strip()
                    contagem = temp_series.value_counts()
                    total_participantes = len(df_filtrado)
        
                    ordem_desejada = ['não', 'sim']
                    contagem_ordenada = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem_ordenada.index]
        
                    if total_participantes == 0 or contagem_ordenada.empty:
                        st.info("Nenhum dado para exibir para Interação Social Voluntária com os filtros selecionados.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels_formatados, contagem_ordenada.values, color='#ff9934', edgecolor='#ff9934')
        
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentual = (height / total_participantes) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                    f'{percentual:.1f}%', # Apenas porcentagem
                                    ha='center', va='bottom', fontsize=9, color='black')
        
                        ax.set_ylim(0, contagem_ordenada.max() * 1.15)
                        ax.set_title('Interage socialmente em diferentes lugares') # Título mais conciso
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Frequência')
                        ax.tick_params(axis='x', rotation=0, labelsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning(f"Coluna '{coluna_interacao_social}' não encontrada ou DataFrame vazio para esta seleção.")

    # --- Gráfico 6: Tratado com Respeito, Dignidade e Igualdade ---
            with col_respeito: # Esta coluna foi definida para ocupar uma linha inteira
                st.subheader(" Olhar do Cuidador Quanto a Tratamento com Respeito, Dignidade e Igualdade")
                coluna_respeito = 'Você considera que o participante é tratado com respeito, dignidade e igualdade pelas outras pessoas'

                if coluna_respeito in df_filtrado.columns and not df_filtrado.empty:
                    temp_series = df_filtrado[coluna_respeito].astype(str).str.lower().str.strip()
                    contagem = temp_series.value_counts()
                    total_participantes = len(df_filtrado)
        
                    ordem_desejada = ['não', 'sim']
                    contagem_ordenada = contagem.reindex(ordem_desejada).dropna()
                    labels_formatados = [s.capitalize() for s in contagem_ordenada.index]
        
                    if total_participantes == 0 or contagem_ordenada.empty:
                        st.info("Nenhum dado para exibir para Tratamento com Respeito com os filtros selecionados.")
                    else:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        bars = ax.bar(labels_formatados, contagem_ordenada.values, color='#680a1d', edgecolor='#680a1d')
        
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            percentual = (height / total_participantes) * 100
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                                    f'{percentual:.1f}%', # Apenas porcentagem
                                    ha='center', va='bottom', fontsize=9, color='black')
        
                        ax.set_ylim(0, contagem_ordenada.max() * 1.15)
                        ax.set_title('Participante é tratado com respeito, dignidade e igualdade') # Título mais conciso
                        ax.set_xlabel('Resposta')
                        ax.set_ylabel('Frequência')
                        ax.tick_params(axis='x', rotation=0, labelsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)
                        plt.tight_layout()
                        for spine in ax.spines.values():
                            spine.set_visible(True)
                            spine.set_linewidth(1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                else:
                    st.warning(f"Coluna '{coluna_respeito}' não encontrada ou DataFrame vazio para esta seleção.")

        # ---------------------
        # --- ABA 7: Correlações (Heatmap com estilo visual harmonizado) ---
        with tab7:
            st.header("Correlações")
            st.markdown("""
            <div class="justified-text">
            Esta seção apresenta as correlações entre as principais variáveis do estudo. 
            A matriz de correlação de Spearman é utilizada para identificar relações significativas entre 
            características sociodemográficas, educacionais e de saúde, sendo mais robusta para dados não-paramétricos e ordinais.
            </div>
            """, unsafe_allow_html=True)
            
            # --- Heatmap 1: Socioeconômicas ---
            st.subheader("Correlação de Variáveis Socioeconômicas")
            try:
                # Criar uma cópia para evitar modificar o DataFrame original fora do escopo da função
                df_corr = df_original.copy()
                
                # --- Tratamento de NaN e Substituições ---
                df_corr.replace('Não sabe', np.nan, inplace=True)
                df_corr.replace('NS', np.nan, inplace=True)
                df_corr.replace('nao sabe', np.nan, inplace=True)
                df_corr.replace('Não sabe o que é', np.nan, inplace=True)
                df_corr.replace('sim', 'Sim', inplace=True)
                df_corr.replace('nao', 'Não', inplace=True)
                df_corr.replace('não', 'Não', inplace=True)

                # --- Conversão de tipos de dados ---
                # A coluna 'Idade do participante' já deve ser numérica do load_data, mas garantimos aqui
                if 'Idade do participante' in df_corr.columns:
                    df_corr['Idade do participante'] = pd.to_numeric(df_corr['Idade do participante'], errors='coerce')

                # 'Quantos irmãos o participante possui' - converter para numérico
                if 'Quantos irmãos o participante possui' in df_corr.columns:
                    df_corr['Quantos irmãos o participante possui'] = pd.to_numeric(df_corr['Quantos irmãos o participante possui'], errors='coerce')
                else:
                    st.warning("Coluna 'Quantos irmãos o participante possui' não encontrada para correlação.")
                    df_corr['Quantos irmãos o participante possui'] = np.nan # Garante que a coluna existe para o próximo passo

                # --- Aplicar codificação para variáveis categóricas que você quer na correlação ---

                # Sexo do participante (Label Encoding)
                if 'Sexo do participante' in df_corr.columns:
                    le_sexo = LabelEncoder()
                    df_corr['Sexo do participante_encoded'] = df_corr['Sexo do participante'].fillna('Não informado').astype(str)
                    df_corr['Sexo do participante_encoded'] = le_sexo.fit_transform(df_corr['Sexo do participante_encoded'])
                else:
                    st.warning("Coluna 'Sexo do participante' não encontrada para correlação.")
                    df_corr['Sexo do participante_encoded'] = np.nan

                # Cor/etnia do participante (Label Encoding)
                if 'Cor/etnia do participante' in df_corr.columns:
                    le_cor_etnia = LabelEncoder()
                    df_corr['Cor/etnia do participante_encoded'] = df_corr['Cor/etnia do participante'].fillna('Não informado').astype(str)
                    df_corr['Cor/etnia do participante_encoded'] = le_cor_etnia.fit_transform(df_corr['Cor/etnia do participante_encoded'])
                else:
                    st.warning("Coluna 'Cor/etnia do participante' não encontrada para correlação.")
                    df_corr['Cor/etnia do participante_encoded'] = np.nan

                # Renda familiar (Ordinal Encoding)
                if 'Renda familiar' in df_corr.columns:
                    # **ATENÇÃO**: Adapte as categorias e a ordem para o seu caso real!
                    # Certifique-se de que todas as categorias presentes na sua coluna estão aqui,
                    # caso contrário, elas serão tratadas como 'unknown_value'
                    categorias_renda = [
                        'Até 1 salário mínimo',
                        '1 a 2 salários mínimos',
                        '2 a 3 salários mínimos',
                        '3 a 4 salários mínimos',
                        '4 a 5 salários mínimos',
                        'Mais de 5 salários mínimos'
                    ]
                    # Adicionando uma categoria para NaN ou valores não esperados, para evitar erros
                    df_corr['Renda familiar_temp'] = df_corr['Renda familiar'].fillna('Missing').astype(str)
                    oe_renda = OrdinalEncoder(categories=[categorias_renda], handle_unknown='use_encoded_value', unknown_value=-1)
                    df_corr['Renda familiar_encoded'] = oe_renda.fit_transform(df_corr[['Renda familiar_temp']])
                else:
                    st.warning("Coluna 'Renda familiar' não encontrada para correlação.")
                    df_corr['Renda familiar_encoded'] = np.nan
                
                # Recebe Bolsa família (Label Encoding)
                if 'Recebe Bolsa família' in df_corr.columns:
                    le_bolsa = LabelEncoder()
                    df_corr['Recebe Bolsa família_encoded'] = df_corr['Recebe Bolsa família'].fillna('Não informado').astype(str)
                    df_corr['Recebe Bolsa família_encoded'] = le_bolsa.fit_transform(df_corr['Recebe Bolsa família_encoded'])
                else:
                    st.warning("Coluna 'Recebe Bolsa família' não encontrada para correlação.")
                    df_corr['Recebe Bolsa família_encoded'] = np.nan

                # Recebe BPC (Label Encoding)
                if 'Recebe BPC' in df_corr.columns:
                    le_bpc = LabelEncoder()
                    df_corr['Recebe BPC_encoded'] = le_bpc.fit_transform(df_corr['Recebe BPC'].fillna('Não informado').astype(str))
                else:
                    st.warning("Coluna 'Recebe BPC' não encontrada para correlação.")
                    df_corr['Recebe BPC_encoded'] = np.nan

                # A residência é (Label Encoding)
                if 'A residência é' in df_corr.columns:
                    le_residencia = LabelEncoder()
                    df_corr['A residência é_encoded'] = le_residencia.fit_transform(df_corr['A residência é'].fillna('Não informado').astype(str))
                else:
                    st.warning("Coluna 'A residência é' não encontrada para correlação.")
                    df_corr['A residência é_encoded'] = np.nan
                
                # 'O participante é alfabetizado' (Label Encoding)
                if 'O participante é alfabetizado' in df_corr.columns:
                    le_alfabetizado = LabelEncoder()
                    df_corr['O participante é alfabetizado_encoded'] = df_corr['O participante é alfabetizado'].fillna('Não informado').astype(str)
                    df_corr['O participante é alfabetizado_encoded'] = le_alfabetizado.fit_transform(df_corr['O participante é alfabetizado_encoded'])
                else:
                    st.warning("Coluna 'O participante é alfabetizado' não encontrada para correlação.")
                    df_corr['O participante é alfabetizado_encoded'] = np.nan


                # --- Seleção das variáveis para a matriz de correlação ---
                variaveis_correlacao = [
                    'Idade do participante', # Adicionei idade, se for interessante
                    'Sexo do participante_encoded',
                    'Cor/etnia do participante_encoded',
                    'Quantos irmãos o participante possui',
                    'Renda familiar_encoded',
                    'Recebe Bolsa família_encoded',
                    'Recebe BPC_encoded',
                    'A residência é_encoded',
                    'O participante é alfabetizado_encoded' # Adicionei alfabetização
                ]

                # Filtrar o DataFrame para apenas as colunas que existem após o pré-processamento
                variaveis_existentes = [col for col in variaveis_correlacao if col in df_corr.columns]

                if not variaveis_existentes:
                    st.warning("Nenhuma das variáveis especificadas para correlação socioeconômica foi encontrada no DataFrame. Verifique os nomes das colunas ou se elas foram criadas após o tratamento.")
                else:
                    # Remover linhas com valores NaN para o cálculo da correlação SOMENTE para as colunas selecionadas
                    df_to_corr = df_corr[variaveis_existentes].dropna()

                    if df_to_corr.empty:
                        st.warning("DataFrame para correlação socioeconômica está vazio após remover NaNs nas colunas selecionadas. Não é possível calcular a correlação.")
                    else:
                        # --- Cálculo da Matriz de Correlação de Spearman ---
                        matriz_correlacao = df_to_corr.corr(method='spearman')

                        # --- Geração do Heatmap da Matriz de Correlação com vmin e vmax ---
                        fig_socio, ax_socio = plt.subplots(figsize=(12, 10))
                        sns.heatmap(
                            matriz_correlacao,
                            annot=True,
                            cmap='coolwarm',
                            fmt=".2f",
                            linewidths=.5,
                            vmin=-1,  # Define o valor mínimo para a escala de cores como -1
                            vmax=1,   # Define o valor máximo para a escala de cores como 1
                            ax=ax_socio
                        )
                        ax_socio.set_title('Matriz de Correlação (Spearman) - Variáveis Socioeconômicas', pad=20)
                        st.pyplot(fig_socio, use_container_width=True)
                        plt.close(fig_socio) # Sempre feche a figura Matplotlib após exibir no Streamlit

                        # Explicação das correlações
                        st.markdown("""
                        
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erro ao calcular ou plotar correlações socioeconômicas: {str(e)}")
                st.warning("""
                Ocorreu um erro ao processar os dados para análise de correlação socioeconômica. 
                Verifique se existem variáveis numéricas ou categóricas codificáveis no conjunto de dados, 
                e se os nomes das colunas estão corretos.
                """)
                # Mostrar exemplo com dados simulados para fins ilustrativos
                st.info("Exemplo ilustrativo de um heatmap de correlação:")
                example_data = pd.DataFrame(np.random.rand(10, 5), columns=['Var A', 'Var B', 'Var C', 'Var D', 'Var E'])
                fig_ex = plt.figure(figsize=(10, 8))
                sns.heatmap(example_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Exemplo de Matriz de Correlação')
                st.pyplot(fig_ex)
                plt.close(fig_ex)

            st.markdown("---") # Separador visual entre os heatmaps

            # --- Heatmap 2: Escolaridade e Alfabetização ---
            st.subheader("Correlação de Escolaridade e Alfabetização")
            try:
                df_edu_corr = df_original.copy()

                # Substituições e tratamento de NaN para este subconjunto de dados
                df_edu_corr.replace('Não sabe', np.nan, inplace=True)
                df_edu_corr.replace('NS', np.nan, inplace=True)
                df_edu_corr.replace('nao sabe', np.nan, inplace=True)
                df_edu_corr.replace('Não sabe o que é', np.nan, inplace=True)
                df_edu_corr.replace('sim', 'Sim', inplace=True)
                df_edu_corr.replace('nao', 'Não', inplace=True)
                df_edu_corr.replace('não', 'Não', inplace=True)

                # --- 8. Codificação de Nível de Escolaridade (OrdinalEncoder) ---
                categorias_escolaridade = [
                    'Não se alfabetizou e/ou não frequentou a escola',
                    'Ensino em escola especial',
                    'Ensino infantil (pré-escola)',
                    'Ensino fundamental incompleto',
                    'Ensino fundamental completo',
                    'Ensino médio incompleto',
                    'Ensino médio completo',
                    'Ensino superior incompleto',
                    'Ensino superior completo',
                    'Pós-graduação'
                ]

                if 'Nível de escolaridade do participante' in df_edu_corr.columns:
                    oe_escolaridade_part = OrdinalEncoder(categories=[categorias_escolaridade], handle_unknown='use_encoded_value', unknown_value=-1)
                    df_edu_corr['Nivel_escolaridade_participante_encoded'] = df_edu_corr['Nível de escolaridade do participante'].fillna('Missing').astype(str)
                    df_edu_corr['Nivel_escolaridade_participante_encoded'] = oe_escolaridade_part.fit_transform(df_edu_corr[['Nivel_escolaridade_participante_encoded']])
                else:
                    st.warning("Aviso: Coluna 'Nível de escolaridade do participante' não encontrada para correlação de educação.")
                    df_edu_corr['Nivel_escolaridade_participante_encoded'] = np.nan

                if 'Nível de escolaridade do responsável do participante' in df_edu_corr.columns:
                    oe_escolaridade_resp = OrdinalEncoder(categories=[categorias_escolaridade], handle_unknown='use_encoded_value', unknown_value=-1)
                    df_edu_corr['Nivel_escolaridade_responsavel_encoded'] = df_edu_corr['Nível de escolaridade do responsável do participante'].fillna('Missing').astype(str)
                    df_edu_corr['Nivel_escolaridade_responsavel_encoded'] = oe_escolaridade_resp.fit_transform(df_edu_corr[['Nivel_escolaridade_responsavel_encoded']])
                else:
                    st.warning("Aviso: Coluna 'Nível de escolaridade do responsável do participante' não encontrada para correlação de educação.")
                    df_edu_corr['Nivel_escolaridade_responsavel_encoded'] = np.nan

                # --- 9. Codificação de variáveis binárias/de habilidade (LabelEncoder) ---
                variaveis_binarias_habilidade = [
                    'O participante é alfabetizado',
                    'Sabe ler',
                    'Sabe escrever',
                    'O participante consegue interpretar texto?'
                ]

                for col in variaveis_binarias_habilidade:
                    # Normaliza o nome da coluna para evitar problemas com caracteres especiais e espaços
                    encoded_col_name = col.replace('?', '').replace(' ', '_').replace('é', 'e').replace('ç', 'c').lower() + '_encoded'
                    if col in df_edu_corr.columns:
                        le = LabelEncoder()
                        df_edu_corr[encoded_col_name] = df_edu_corr[col].fillna('Não informado').astype(str)
                        df_edu_corr[encoded_col_name] = le.fit_transform(df_edu_corr[encoded_col_name])
                    else:
                        st.warning(f"Aviso: Coluna '{col}' não encontrada para correlação de educação.")
                        df_edu_corr[encoded_col_name] = np.nan


                # --- 10. Seleção das variáveis para a matriz de correlação ---
                variaveis_para_corr_educacao = [
                    'Nivel_escolaridade_participante_encoded',
                    'Nivel_escolaridade_responsavel_encoded',
                    'o_participante_e_alfabetizado_encoded',
                    'sabe_ler_encoded',
                    'sabe_escrever_encoded',
                    'o_participante_consegue_interpretar_texto_encoded'
                ]

                # Filtrar o DataFrame para apenas as colunas que existem no df_edu_corr final
                df_corr_educacao_filtered = df_edu_corr[[col for col in variaveis_para_corr_educacao if col in df_edu_corr.columns]].copy()

                # Remover linhas com valores NaN (para qualquer uma das colunas selecionadas)
                df_corr_educacao_filtered.dropna(inplace=True)

                # --- 11. Gerar o Heatmap da Matriz de Correlação (Educação/Alfabetização) ---
                if df_corr_educacao_filtered.empty:
                    st.warning("DataFrame para correlação de educação/alfabetização está vazio após remover NaNs. Não é possível calcular a correlação.")
                else:
                    # Mapeamento para nomes de variáveis mais curtos no gráfico
                    short_labels_map = {
                        'Nivel_escolaridade_participante_encoded': 'Esc. Part.',
                        'Nivel_escolaridade_responsavel_encoded': 'Esc. Resp.',
                        'o_participante_e_alfabetizado_encoded': 'Alfabetizado',
                        'sabe_ler_encoded': 'Sabe Ler',
                        'sabe_escrever_encoded': 'Sabe Escrever',
                        'o_participante_consegue_interpretar_texto_encoded': 'Interpreta Texto'
                    }

                    # Aplicar o mapeamento à matriz de correlação
                    matriz_correlacao_educacao = df_corr_educacao_filtered.corr(method='spearman')
                    matriz_correlacao_educacao.columns = [short_labels_map.get(col, col) for col in matriz_correlacao_educacao.columns]
                    matriz_correlacao_educacao.index = [short_labels_map.get(idx, idx) for idx in matriz_correlacao_educacao.index]

                    # Ajusta o tamanho da figura dinamicamente e de forma mais generosa
                    num_vars_educacao = len(matriz_correlacao_educacao.columns)
                    fig_edu, ax_edu = plt.subplots(figsize=(num_vars_educacao * 1.5, num_vars_educacao * 1.3)) # Fatores aumentados

                    sns.heatmap(
                        matriz_correlacao_educacao,
                        annot=True,
                        cmap='coolwarm',
                        fmt=".2f",
                        linewidths=.5,
                        vmin=-1,
                        vmax=1,
                        cbar_kws={'shrink': 0.8},
                        annot_kws={"fontsize": 10}, # Ajusta o tamanho da fonte das anotações
                        ax=ax_edu
                    )
                    ax_edu.set_title('Matriz de Correlação (Spearman) - Escolaridade e Alfabetização', fontsize=14)
                    
                    # Correção para o erro set_ticks()
                    ax_edu.set_xticklabels(ax_edu.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                    ax_edu.set_yticklabels(ax_edu.get_yticklabels(), rotation=0, fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig_edu, use_container_width=True)
                    plt.close(fig_edu)

                    st.markdown("""
                
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erro ao calcular ou plotar correlações de educação/alfabetização: {str(e)}")
                st.warning("""
                Ocorreu um erro ao processar os dados para análise de correlação de educação/alfabetização. 
                Verifique se existem variáveis relevantes e se os nomes das colunas estão corretos.
                """)
                # Mostrar exemplo com dados simulados para fins ilustrativos
                st.info("Exemplo ilustrativo de um heatmap de correlação:")
                example_data_edu = pd.DataFrame(np.random.rand(10, 3), columns=['Nível Esc.', 'Sabe Ler', 'Interpreta'])
                fig_ex_edu = plt.figure(figsize=(8, 6))
                sns.heatmap(example_data_edu.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Exemplo de Matriz de Correlação (Educação)')
                st.pyplot(fig_ex_edu)
                plt.close(fig_ex_edu)

            st.markdown("---") # Separador visual entre os heatmaps

            # --- Heatmap 4: Aspectos Demográficos e Saúde (o novo heatmap) ---
            st.subheader("Correlação de Variáveis Demográficas e de Saúde")
            st.markdown("""
            """, unsafe_allow_html=True)
            try:
                # Criar uma cópia limpa do DataFrame original para este heatmap específico
                df_demog_saude_corr = df_original.copy()

                # --- 2. Tratamento e limpeza de dados iniciais (global) ---
                df_demog_saude_corr.replace(['Não sabe', 'NS', 'nao sabe', 'Não sabe o que é'], np.nan, inplace=True)
                df_demog_saude_corr.replace('sim', 'Sim', inplace=True)
                df_demog_saude_corr.replace(['nao', 'não'], 'Não', inplace=True)

                # --- 3. Processamento e Codificação das VARIÁVEIS PARA O HEATMAP ---

                # Renda familiar (OrdinalEncoder)
                if 'Renda familiar' in df_demog_saude_corr.columns:
                    categorias_renda = [
                        'Até 1 salário mínimo',
                        '1 a 2 salários mínimos',
                        '2 a 3 salários mínimos',
                        '3 a 4 salários mínimos',
                        '4 a 5 salários mínimos',
                        'Mais de 5 salários mínimos'
                    ]
                    oe_renda = OrdinalEncoder(categories=[categorias_renda], handle_unknown='use_encoded_value', unknown_value=-1)
                    df_demog_saude_corr['Renda_familiar_encoded'] = df_demog_saude_corr['Renda familiar'].fillna('Missing').astype(str)
                    df_demog_saude_corr['Renda_familiar_encoded'] = oe_renda.fit_transform(df_demog_saude_corr[['Renda_familiar_encoded']])
                else:
                    st.warning("Aviso: Coluna 'Renda familiar' não encontrada. Será preenchida com NaN.")
                    df_demog_saude_corr['Renda_familiar_encoded'] = np.nan

                # Nível de escolaridade do participante (OrdinalEncoder)
                if 'Nível de escolaridade do participante' in df_demog_saude_corr.columns:
                    categorias_escolaridade = [
                        'Não se alfabetizou e/ou não frequentou a escola',
                        'Ensino em escola especial',
                        'Ensino infantil (pré-escola)',
                        'Ensino fundamental incompleto',
                        'Ensino fundamental completo',
                        'Ensino médio incompleto',
                        'Ensino médio completo',
                        'Ensino superior incompleto',
                        'Ensino superior completo',
                        'Pós-graduação'
                    ]
                    oe_escol_participante = OrdinalEncoder(categories=[categorias_escolaridade], handle_unknown='use_encoded_value', unknown_value=-1)
                    df_demog_saude_corr['Nivel_escolaridade_participante_encoded'] = df_demog_saude_corr['Nível de escolaridade do participante'].fillna('Missing').astype(str)
                    df_demog_saude_corr['Nivel_escolaridade_participante_encoded'] = oe_escol_participante.fit_transform(df_demog_saude_corr[['Nivel_escolaridade_participante_encoded']])
                else:
                    st.warning("Aviso: Coluna 'Nível de escolaridade do participante' não encontrada. Será preenchida com NaN.")
                    df_demog_saude_corr['Nivel_escolaridade_participante_encoded'] = np.nan

                # Nível de escolaridade do responsável do participante (OrdinalEncoder)
                if 'Nível de escolaridade do responsável do participante' in df_demog_saude_corr.columns:
                    oe_escol_responsavel = OrdinalEncoder(categories=[categorias_escolaridade], handle_unknown='use_encoded_value', unknown_value=-1)
                    df_demog_saude_corr['Nivel_escolaridade_responsavel_encoded'] = df_demog_saude_corr['Nível de escolaridade do responsável do participante'].fillna('Missing').astype(str)
                    df_demog_saude_corr['Nivel_escolaridade_responsavel_encoded'] = oe_escol_responsavel.fit_transform(df_demog_saude_corr[['Nivel_escolaridade_responsavel_encoded']])
                else:
                    st.warning("Aviso: Coluna 'Nível de escolaridade do responsável do participante' não encontrada. Será preenchida com NaN.")
                    df_demog_saude_corr['Nivel_escolaridade_responsavel_encoded'] = np.nan


                # Idade em que a mãe do participante teve a gestação (Numérica)
                if 'Idade em que a mãe do participante teve a gestação' in df_demog_saude_corr.columns:
                    df_demog_saude_corr['Idade_mae_gestacao'] = pd.to_numeric(df_demog_saude_corr['Idade em que a mãe do participante teve a gestação'], errors='coerce')
                else:
                    st.warning("Aviso: Coluna 'Idade em que a mãe do participante teve a gestação' não encontrada. Será preenchida com NaN.")
                    df_demog_saude_corr['Idade_mae_gestacao'] = np.nan

                # Realizou o acompanhamento pré natal na rede (One-Hot Encoding)
                if 'Realizou o acompanhamento pré natal na rede:' in df_demog_saude_corr.columns:
                    df_demog_saude_corr['Pre_natal_rede_temp'] = df_demog_saude_corr['Realizou o acompanhamento pré natal na rede:'].fillna('Não informado').astype(str)
                    df_demog_saude_corr = pd.get_dummies(df_demog_saude_corr, columns=['Pre_natal_rede_temp'], prefix='PreNatalRede', dummy_na=False)
                else:
                    st.warning("Aviso: Coluna 'Realizou o acompanhamento pré natal na rede:' não encontrada. Nenhuma coluna de pré-natal será criada.")

                # Cor/etnia do participante (One-Hot Encoding)
                if 'Cor/etnia do participante' in df_demog_saude_corr.columns:
                    df_demog_saude_corr['Cor_etnia_participante_temp'] = df_demog_saude_corr['Cor/etnia do participante'].fillna('Não informado').astype(str)
                    df_demog_saude_corr = pd.get_dummies(df_demog_saude_corr, columns=['Cor_etnia_participante_temp'], prefix='CorEtnia', dummy_na=False)
                else:
                    st.warning("Aviso: Coluna 'Cor/etnia do participante' não encontrada.")

                # Pratica alguma atividade física ou esporte (LabelEncoder)
                if 'Pratica alguma atividade física ou esporte' in df_demog_saude_corr.columns:
                    le_atividade_fisica = LabelEncoder()
                    df_demog_saude_corr['Atividade_fisica_encoded'] = df_demog_saude_corr['Pratica alguma atividade física ou esporte'].fillna('Não informado').astype(str)
                    df_demog_saude_corr['Atividade_fisica_encoded'] = le_atividade_fisica.fit_transform(df_demog_saude_corr['Atividade_fisica_encoded'])
                else:
                    st.warning("Aviso: Coluna 'Pratica alguma atividade física ou esporte' não encontrada. Será preenchida com NaN.")
                    df_demog_saude_corr['Atividade_fisica_encoded'] = np.nan

                # --- 4. Seleção das variáveis para a matriz de correlação ---
                variaveis_para_corr_demog_saude = [
                    'Renda_familiar_encoded',
                    'Nivel_escolaridade_participante_encoded',
                    'Nivel_escolaridade_responsavel_encoded',
                    'Idade_mae_gestacao',
                    'Atividade_fisica_encoded'
                ]

                # Adicionar as colunas one-hot encoded de Cor/Etnia e Pré-Natal criadas dinamicamente
                colunas_coretnia_dummy_demog = [col for col in df_demog_saude_corr.columns if col.startswith('CorEtnia_')]
                colunas_prenatal_dummy_demog = [col for col in df_demog_saude_corr.columns if col.startswith('PreNatalRede_')]
                variaveis_para_corr_demog_saude.extend(colunas_coretnia_dummy_demog)
                variaveis_para_corr_demog_saude.extend(colunas_prenatal_dummy_demog)

                # Filtrar o DataFrame para apenas as colunas que realmente existem após o processamento
                df_corr_demog_saude = df_demog_saude_corr[[col for col in variaveis_para_corr_demog_saude if col in df_demog_saude_corr.columns]].copy()
                
                # NOVO: Verificar e remover colunas com apenas NaNs ou variância zero
                cols_to_drop_demog_saude = []
                for col in df_corr_demog_saude.columns:
                    is_all_null = df_corr_demog_saude[col].isnull().all()
                    has_zero_variance = df_corr_demog_saude[col].nunique() <= 1

                    if is_all_null:
                        st.warning(f"Aviso: Coluna '{col}' contém apenas NaNs e será removida do cálculo da correlação para o heatmap de demográficos/saúde.")
                        cols_to_drop_demog_saude.append(col)
                    elif has_zero_variance and not is_all_null:
                        st.warning(f"Aviso: Coluna '{col}' tem variância zero (todos os valores são iguais) e será removida do cálculo da correlação para o heatmap de demográficos/saúde.")
                        cols_to_drop_demog_saude.append(col)

                if cols_to_drop_demog_saude:
                    df_corr_demog_saude.drop(columns=cols_to_drop_demog_saude, inplace=True)
                
                # Remover linhas com valores NaN restantes para o cálculo da correlação
                df_corr_demog_saude.dropna(inplace=True)

                # --- 5. Gerar o Heatmap da Matriz de Correlação ---
                if df_corr_demog_saude.empty:
                    st.warning("DataFrame para correlação de aspectos demográficos e saúde está vazio após remover NaNs. Não é possível calcular a correlação. Verifique se há dados suficientes nas variáveis selecionadas.")
                else:
                    # Mapeamento para nomes de variáveis mais curtos no gráfico
                    short_labels_map_demog_saude = {
                        'Renda_familiar_encoded': 'Renda Familiar',
                        'Nivel_escolaridade_participante_encoded': 'Escolaridade Part.',
                        'Nivel_escolaridade_responsavel_encoded': 'Escolaridade Resp.',
                        'Idade_mae_gestacao': 'Idade Mãe Gest.',
                        'Atividade_fisica_encoded': 'Ativ. Física',
                        # Mapeamentos para One-Hot Encoding de Cor/Etnia
                        'CorEtnia_Branca': 'Etnia Branca',
                        'CorEtnia_Preta': 'Etnia Preta',
                        'CorEtnia_Parda': 'Etnia Parda',
                        'CorEtnia_Amarela': 'Etnia Amarela',
                        'CorEtnia_Indígena': 'Etnia Indígena',
                        'CorEtnia_Não informado': 'Etnia Não Info',
                        # Mapeamentos para One-Hot Encoding de Pré-Natal
                        'PreNatalRede_privada': 'Pré-Natal Priv.',
                        'PreNatalRede_publica': 'Pré-Natal Púb.',
                        'PreNatalRede_não realizou': 'Pré-Natal Não Real.',
                        'PreNatalRede_Não informado': 'Pré-Natal Não Info',
                    }

                    # Calcula a matriz de correlação de Spearman
                    matriz_correlacao_demog_saude = df_corr_demog_saude.corr(method='spearman')

                    # Aplica o mapeamento aos rótulos da matriz
                    matriz_correlacao_demog_saude.columns = [short_labels_map_demog_saude.get(col, col) for col in matriz_correlacao_demog_saude.columns]
                    matriz_correlacao_demog_saude.index = [short_labels_map_demog_saude.get(idx, idx) for idx in matriz_correlacao_demog_saude.index]

                    # Ajusta o tamanho da figura e fontes para melhor visualização
                    num_vars_demog_saude = len(matriz_correlacao_demog_saude.columns)
                    fig_demog_saude, ax_demog_saude = plt.subplots(figsize=(num_vars_demog_saude * 0.9, num_vars_demog_saude * 0.8))

                    sns.heatmap(
                        matriz_correlacao_demog_saude,
                        annot=True,
                        cmap='coolwarm',
                        fmt=".2f",
                        linewidths=.5,
                        vmin=-1,
                        vmax=1,
                        cbar_kws={'shrink': 0.8},
                        annot_kws={"fontsize": 7},
                        ax=ax_demog_saude
                    )
                    ax_demog_saude.set_title('Matriz de Correlação (Spearman) - Aspectos Demográficos e Saúde', fontsize=12)
                    
                    # Correção para o erro set_ticks()
                    ax_demog_saude.set_xticklabels(ax_demog_saude.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                    ax_demog_saude.set_yticklabels(ax_demog_saude.get_yticklabels(), rotation=0, fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig_demog_saude, use_container_width=True)
                    plt.close(fig_demog_saude)

    


            except Exception as e:
                st.error(f"Erro ao calcular ou plotar correlações demográficas e de saúde: {str(e)}")
                st.warning("""
                Ocorreu um erro ao processar os dados para análise de correlação de aspectos demográficos e saúde. 
                Verifique se existem variáveis relevantes e se os nomes das colunas estão corretos, especialmente para One-Hot Encoding.
                """)
                st.info("Exemplo ilustrativo de um heatmap de correlação:")
                example_data_demog_saude = pd.DataFrame(np.random.rand(10, 5), columns=['Renda', 'Esc. Part.', 'IMC', 'Etnia', 'Ativ. Física'])
                fig_ex_demog_saude = plt.figure(figsize=(10, 8))
                sns.heatmap(example_data_demog_saude.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Exemplo de Matriz de Correlação (Demográficos e Saúde)')
                st.pyplot(fig_ex_demog_saude)
                plt.close(fig_ex_demog_saude)   

            st.markdown("---") # Separador visual entre os heatmaps

        # --- ABA 8: PCA e Cluster ---
        with tab8:
            st.header("PCA e Cluster")
            
            st.markdown("""
            <div class="justified-text">
            Esta seção apresenta a análise de redução de dimensionalidade (PCA) e os 
            resultados da clusterização dos dados. Essas técnicas ajudam a identificar 
            padrões e agrupamentos naturais nos dados.
            </div>
            """, unsafe_allow_html=True)
            
            # Análise PCA
            st.subheader("Análise de Componentes Principais (PCA)")
            # ======================
            # SELEÇÃO DE VARIÁVEIS
            # ======================
            variaveis = [
                'Idade do participante',
                'Idade do cuidador principal',
                'IMC',
                'Peso do participante', 
                'Altura do participante',
                'Idade em que a mãe do participante teve a gestação'
            ]
            
            # ======================
            # PRÉ-PROCESSAMENTO
            # ======================
            with st.expander("Pré-processamento dos Dados", expanded=True):
                # Carregar dados (substitua por seu DataFrame)
                
                # Preencher NA com medianas
                df_para_analise = df_original[variaveis].copy()
                for col in variaveis:
                    df_para_analise[col] = df_para_analise[col].fillna(df_para_analise[col].median())
                
                st.success(f"Dados pré-processados com sucesso! Total de participantes: {len(df_para_analise)}")
                
                # Mostrar estatísticas básicas
                if st.checkbox("Mostrar estatísticas descritivas"):
                    st.dataframe(df_para_analise.describe())
            
            # ======================
            # ANÁLISE PCA E CLUSTERS
            # ======================
            with st.expander("Análise de Clusters", expanded=True):
                # Widget para selecionar número de clusters
                n_clusters = st.slider("Número de Clusters", 2, 5, 3)
                
                # Normalização e PCA
                scaler = StandardScaler()
                dados_normalizados = scaler.fit_transform(df_para_analise)
                
                pca = PCA(n_components=2)
                componentes = pca.fit_transform(dados_normalizados)
                var_exp = pca.explained_variance_ratio_ * 100
                
                # K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(componentes)
                df_para_analise['Cluster'] = clusters
                
                # ======================
                # VISUALIZAÇÕES
                # ======================
                st.subheader("Visualização dos Clusters (PCA)")
                fig1, ax1 = plt.subplots(figsize=(8,6))
                sns.scatterplot(
                    x=componentes[:, 0], y=componentes[:, 1],
                    hue=clusters, palette='Set2', s=100, ax=ax1
                )
                ax1.set_title(f'Clusters via PCA + K-Means (k={n_clusters})')
                ax1.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variância)')
                ax1.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variância)')
                ax1.grid(True)
                st.pyplot(fig1)

                st.subheader("Médias por Cluster")
                fig2, ax2 = plt.subplots(figsize=(10,6))
                df_melted = df_para_analise.melt(
                    id_vars='Cluster', 
                    value_vars=variaveis,
                    var_name='Variável', 
                    value_name='Valor'
                )
                sns.barplot(
                    data=df_melted, 
                    x='Variável', 
                    y='Valor', 
                    hue='Cluster', 
                    palette='Set2', 
                    errorbar=None,
                    ax=ax2
                )
                ax2.set_title('Médias das Variáveis por Cluster')
                ax2.set_ylabel('Valor Médio')
                ax2.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig2)

            # ======================
            # ANÁLISE DE ATIVIDADE FÍSICA POR CLUSTER
            # ======================
            with st.expander("Análise de Atividade Física por Cluster", expanded=True):
                # Juntar clusters com dados originais
                df_clusterizado = df_original.copy()
                df_clusterizado['Cluster'] = df_para_analise['Cluster']
                
                # Selecionar coluna de atividade física
                col_atividade = st.selectbox(
                    "Selecione a coluna de atividade física:",
                    options=[col for col in df_original.columns if 'atividade' in col.lower() or 'esporte' in col.lower()],
                    index=0
                )
                
                if col_atividade:
                    # Filtrar apenas casos com informação de atividade física
                    df_atividade = df_clusterizado[df_clusterizado[col_atividade].notna()]
                    
                    # Tabelas cruzadas
                    st.subheader("Distribuição de Atividade Física por Cluster")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Tabela de contagem absoluta
                        tabela_cruzada = pd.crosstab(df_atividade['Cluster'], df_atividade[col_atividade])
                        st.markdown("**Contagem absoluta**")
                        st.dataframe(tabela_cruzada.style.background_gradient(cmap='Blues'))
                    
                    with col2:
                        # Tabela percentual
                        tabela_percentual = tabela_cruzada.div(tabela_cruzada.sum(axis=1), axis=0) * 100
                        st.markdown("**Percentual por cluster (%)**")
                        st.dataframe(tabela_percentual.round(1).style.background_gradient(cmap='Greens'))
                    
                    # Gráfico de barras empilhadas
                    st.subheader("Distribuição Percentual")
                    fig3, ax3 = plt.subplots(figsize=(10,6))
                    tabela_percentual.plot(kind='bar', stacked=True, colormap='Set2', ax=ax3)
                    ax3.set_title('Prática de Atividade Física por Cluster')
                    ax3.set_ylabel('% de participantes')
                    ax3.set_xlabel('Cluster')
                    ax3.legend(title=col_atividade, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    st.pyplot(fig3)
                    
                    # Teste Qui-Quadrado (opcional)
                    if st.checkbox("Realizar teste de associação (Qui-Quadrado)"):
                        try:
                            chi2, p_valor, dof, expected = chi2_contingency(tabela_cruzada)
                            
                            st.markdown(f"""
                            **Resultados do Teste Qui-Quadrado:**
                            - Estatística qui-quadrado: `{chi2:.4f}`
                            - p-valor: `{p_valor:.4f}`
                            - Graus de liberdade: `{dof}`
                            """)
                            
                            if p_valor < 0.05:
                                st.success("🔍 **Resultado:** Existe associação significativa entre clusters e atividade física (p < 0.05)")
                            else:
                                st.warning("🔍 **Resultado:** NÃO há associação significativa entre clusters e atividade física (p ≥ 0.05)")
                        
                        except Exception as e:
                            st.error(f"Erro ao calcular o teste qui-quadrado: {str(e)}")
                else:
                    st.warning("Nenhuma coluna de atividade física encontrada. Verifique os nomes das colunas.")
            
            # ======================
            # ESTATÍSTICAS POR CLUSTER
            # ======================
            with st.expander("Estatísticas Detalhadas por Cluster", expanded=False):
                tabs = st.tabs(["Médias", "Medianas", "Tamanhos"])
                
                with tabs[0]:
                    st.dataframe(
                        df_para_analise.groupby('Cluster')[variaveis].mean().style.background_gradient(cmap='Blues')
                    )
                
                with tabs[1]:
                    st.dataframe(
                        df_para_analise.groupby('Cluster')[variaveis].median().style.background_gradient(cmap='Greens')
                    )
                
                with tabs[2]:
                    cluster_counts = df_para_analise['Cluster'].value_counts().sort_index()
                    st.dataframe(cluster_counts)
                    fig4, ax4 = plt.subplots(figsize=(6,4))
                    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='Set2')
                    ax4.set_title('Distribuição dos Clusters')
                    ax4.set_xlabel('Cluster')
                    ax4.set_ylabel('Número de Participantes')
                    st.pyplot(fig4)

            # ======================
            # ANÁLISE DE PERCEPÇÃO DE ATENDIMENTO
            # ======================
            with st.expander("Análise de Percepção de Atendimento de Saúde", expanded=True):
                col_atividade = 'Você considera que o participante possui atendimento de saúde adequado pelo sistema público para atender suas necessidades'
                
                if col_atividade in df_original.columns:
                    # 1. Preparar dados
                    df_clusterizado_cat = df_para_analise.copy()
                    df_clusterizado_cat[col_atividade] = df_original[col_atividade]
                    
                    # 2. Padronizar respostas
                    df_clusterizado_cat[col_atividade] = (
                        df_clusterizado_cat[col_atividade]
                        .fillna('Não')
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .apply(lambda x: 'Sim' if x in ['sim', 'sim.'] else 'Não')
                    )
                    
                    # 3. Tabelas cruzadas
                    st.subheader("Distribuição por Cluster")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        tabela_cruzada = pd.crosstab(df_clusterizado_cat['Cluster'], df_clusterizado_cat[col_atividade])
                        st.markdown("**Contagem absoluta**")
                        st.dataframe(tabela_cruzada.style.background_gradient(cmap='Blues'))
                    
                    with col2:
                        tabela_percentual = tabela_cruzada.div(tabela_cruzada.sum(axis=1), axis=0) * 100
                        st.markdown("**Percentual por cluster (%)**")
                        st.dataframe(tabela_percentual.round(1).style.background_gradient(cmap='Greens'))
                    
                    # 4. Gráficos
                    st.subheader("Visualizações")
                    
                    tab1, tab2 = st.tabs(["Gráfico Empilhado", "Gráfico Lado a Lado"])
                    
                    with tab1:
                        fig3, ax3 = plt.subplots(figsize=(10,6))
                        tabela_percentual[['Sim', 'Não']].plot(kind='bar', stacked=True, colormap='Set2', ax=ax3)
                        ax3.set_title('Percepção de Atendimento de Saúde por Cluster')
                        ax3.set_ylabel('% de participantes')
                        ax3.set_xlabel('Cluster')
                        ax3.legend(title='Resposta', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        st.pyplot(fig3)
                    
                    with tab2:
                        fig4, ax4 = plt.subplots(figsize=(10,6))
                        tabela_percentual[['Sim', 'Não']].plot(kind='bar', stacked=False, color=['#66c2a5', '#fc8d62'], ax=ax4)
                        ax4.set_title('Percepção de Atendimento de Saúde por Cluster')
                        ax4.set_ylabel('% de participantes')
                        ax4.set_xlabel('Cluster')
                        ax4.legend(title='Resposta')
                        ax4.grid(axis='y', linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        st.pyplot(fig4)
                    
                    # 5. Teste Qui-Quadrado
                    st.subheader("Teste de Associação Estatística")
                    chi2, p_valor, dof, expected = chi2_contingency(tabela_cruzada)
                    
                    st.markdown(f"""
                    **Resultados do Teste Qui-Quadrado:**
                    - Estatística qui-quadrado: `{chi2:.4f}`
                    - p-valor: `{p_valor:.4f}`
                    - Graus de liberdade: `{dof}`
                    """)
                    
                    if p_valor < 0.05:
                        st.success("🔍 **Resultado:** Existe diferença estatisticamente significativa entre os clusters na percepção de atendimento (p < 0.05)")
                    else:
                        st.warning("🔍 **Resultado:** NÃO há diferença estatisticamente significativa entre os clusters na percepção de atendimento (p ≥ 0.05)")
                    
                    st.markdown("**Frequências esperadas (se não houvesse associação):**")
                    st.dataframe(pd.DataFrame(expected, index=tabela_cruzada.index, columns=tabela_cruzada.columns))
                    
                else:
                    st.error(f"Coluna '{col_atividade}' não encontrada no DataFrame. Verifique o nome da variável.")
            
            # ======================
            # ESTATÍSTICAS POR CLUSTER
            # ======================
            with st.expander("Estatísticas Detalhadas por Cluster", expanded=False):
                tabs = st.tabs(["Médias", "Medianas", "Tamanhos"])
                
                with tabs[0]:
                    st.dataframe(
                        df_para_analise.groupby('Cluster')[variaveis].mean().style.background_gradient(cmap='Blues')
                    )
                
                with tabs[1]:
                    st.dataframe(
                        df_para_analise.groupby('Cluster')[variaveis].median().style.background_gradient(cmap='Greens')
                    )
                
                with tabs[2]:
                    cluster_counts = df_para_analise['Cluster'].value_counts().sort_index()
                    st.dataframe(cluster_counts)
                    fig5, ax5 = plt.subplots(figsize=(6,4))
                    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='Set2')
                    ax5.set_title('Distribuição dos Clusters')
                    ax5.set_xlabel('Cluster')
                    ax5.set_ylabel('Número de Participantes')
                    st.pyplot(fig5)

        
        # --- ABA 9: Regressão e Random Forest ---
        with tab9:
            st.header("Modelos Preditivos: Regressão e Random Forest")
            
            st.markdown("""
            <div style="text-align: justify">
            Esta seção apresenta modelos preditivos para análise dos dados, incluindo regressão linear 
            para problemas de predição numérica e Random Forest para classificação.
            </div>
            """, unsafe_allow_html=True)
            
            # Função para verificar variáveis válidas para regressão
            def get_valid_regression_targets(df):
                valid_targets = []
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                for col in numeric_cols:
                    # Verificar se tem valores suficientes e variância
                    if df[col].nunique() > 5 and df[col].notna().sum() > 20:
                        valid_targets.append(col)
                
                return valid_targets
            
            # Função para verificar variáveis válidas para classificação
            def get_valid_classification_targets(df):
                valid_targets = []
                
                # Considerar colunas categóricas e binárias numéricas
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                binary_num_cols = [col for col in df.select_dtypes(include=['number']) 
                                  if df[col].nunique() == 2 and df[col].notna().sum() > 20]
                
                for col in cat_cols + binary_num_cols:
                    # Verificar se tem pelo menos 2 classes com amostras suficientes
                    value_counts = df[col].value_counts()
                    if len(value_counts) >= 2 and all(value_counts > 5):
                        valid_targets.append(col)
                
                return valid_targets
            
            # Obter variáveis válidas
            valid_reg_targets = get_valid_regression_targets(df_filtrado)
            valid_clf_targets = get_valid_classification_targets(df_filtrado)
            
            # Mostrar apenas se houver variáveis válidas
            if not valid_reg_targets and not valid_clf_targets:
                st.warning("Nenhuma variável alvo adequada encontrada para modelagem.")
            else:
                # Divisão em duas colunas para seleção de parâmetros
                col1, col2 = st.columns(2)
                
                with col1:
                    if valid_reg_targets:
                        # Selecionar variável alvo para regressão
                        target_reg = st.selectbox(
                            "Selecione a variável alvo para regressão:",
                            options=valid_reg_targets,
                            index=0
                        )
                    else:
                        st.warning("Nenhuma variável numérica adequada para regressão encontrada.")
                
                with col2:
                    if valid_clf_targets:
                        # Selecionar variável alvo para classificação
                        target_clf = st.selectbox(
                            "Selecione a variável alvo para classificação:",
                            options=valid_clf_targets,
                            index=0
                        )
                    else:
                        st.warning("Nenhuma variável categórica adequada para classificação encontrada.")
                
                # Divisão em abas para cada modelo
                tab_reg, tab_clf = st.tabs(["Análise de Regressão", "Random Forest"])
                
                # ABA DE REGRESSÃO
                with tab_reg:
                    if valid_reg_targets:
                        st.subheader(f"Análise de Regressão para: {target_reg}")
                        
                        try:
                            import statsmodels.api as sm
                            from sklearn.preprocessing import LabelEncoder
                            
                            # Preparar dados para regressão
                            df_reg = df_filtrado.copy()
                            
                            # Selecionar features automaticamente (todas as numéricas exceto a target)
                            numeric_features = [col for col in df_filtrado.select_dtypes(include=['number']).columns 
                                              if col != target_reg and df_filtrado[col].notna().sum() > 20]
                            
                            if len(numeric_features) > 0:
                                # Processar dados
                                df_reg = df_reg[[target_reg] + numeric_features].dropna()
                                
                                if len(df_reg) > 30:  # Mínimo de 30 observações
                                    X = df_reg[numeric_features]
                                    y = df_reg[target_reg]
                                    
                                    # Adicionar constante para o intercepto
                                    X = sm.add_constant(X)
                                    
                                    # Ajustar modelo
                                    model = sm.OLS(y, X).fit()
                                    
                                    # Exibir resultados
                                    st.write("### Resumo do Modelo")
                                    
                                    # Criar duas colunas para métricas
                                    m1, m2 = st.columns(2)
                                    with m1:
                                        st.metric("R² Ajustado", f"{model.rsquared_adj:.3f}")
                                    with m2:
                                        st.metric("Valor F", f"{model.fvalue:.1f}", f"p-valor: {model.f_pvalue:.4f}")
                                    
                                    # Mostrar coeficientes em tabela
                                    st.write("### Coeficientes do Modelo")
                                    coef_df = pd.DataFrame({
                                        'Variável': model.params.index,
                                        'Coeficiente': model.params.values,
                                        'p-valor': model.pvalues.values
                                    })
                                    st.dataframe(
                                        coef_df.style.format({'Coeficiente': '{:.4f}', 'p-valor': '{:.4f}'})
                                        .apply(lambda x: ['background-color: #ffcccc' if x['p-valor'] > 0.05 else '' for i in x], axis=1)
                                    )
                                    
                                    # Gráficos de diagnóstico
                                    st.write("### Diagnóstico do Modelo")
                                    
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                    
                                    # Gráfico de resíduos
                                    ax1.scatter(model.predict(), model.resid, alpha=0.6, color='#4e79a7')
                                    ax1.axhline(y=0, color='r', linestyle='--')
                                    ax1.set_title('Resíduos vs Valores Preditos')
                                    ax1.set_xlabel('Valores Preditos')
                                    ax1.set_ylabel('Resíduos')
                                    
                                    # QQ-plot
                                    sm.qqplot(model.resid, line='s', ax=ax2)
                                    ax2.set_title('QQ-Plot dos Resíduos')
                                    
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    
                                else:
                                    st.warning(f"Dados insuficientes após limpeza (apenas {len(df_reg)} observações válidas).")
                            else:
                                st.warning("Nenhuma feature numérica adequada encontrada para a regressão.")
                                
                        except Exception as e:
                            st.error(f"Erro na regressão: {str(e)}")
                    else:
                        st.warning("Nenhuma variável alvo válida selecionada para regressão.")
                
                # ABA DE CLASSIFICAÇÃO
                with tab_clf:
                    if valid_clf_targets:
                        st.subheader(f"Modelo Random Forest para: {target_clf}")
                        
                        try:
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
                            from sklearn.preprocessing import LabelEncoder
                            
                            # Preparar dados
                            df_clf = df_filtrado.copy()
                            
                            # Codificar target
                            le = LabelEncoder()
                            df_clf[target_clf] = le.fit_transform(df_clf[target_clf].astype(str))
                            
                            # Selecionar features (todas as numéricas com dados suficientes)
                            numeric_features = [col for col in df_filtrado.select_dtypes(include=['number']).columns 
                                              if col != target_clf and df_filtrado[col].notna().sum() > 20]
                            
                            if len(numeric_features) > 0:
                                df_clf = df_clf[[target_clf] + numeric_features].dropna()
                                
                                if len(df_clf) > 30:  # Mínimo de 30 observações
                                    X = df_clf[numeric_features]
                                    y = df_clf[target_clf]
                                    
                                    # Verificar balanceamento
                                    class_balance = pd.Series(y).value_counts(normalize=True)
                                    st.write(f"**Distribuição das classes:** {', '.join([f'{le.classes_[i]}: {p:.1%}' for i, p in class_balance.items()])}")
                                    
                                    # Dividir dados com stratificação
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.3, random_state=42, stratify=y
                                    )
                                    
                                    # Configuração do modelo
                                    st.write("### Configuração do Modelo")
                                    n_estimators = st.slider("Número de árvores", 10, 200, 100, key='n_estimators')
                                    max_depth = st.slider("Profundidade máxima", 2, 20, 5, key='max_depth')
                                    
                                    # Treinar modelo
                                    model = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        random_state=42,
                                        class_weight='balanced' if any(p < 0.3 for p in class_balance) else None
                                    )
                                    model.fit(X_train, y_train)
                                    
                                    # Avaliação
                                    y_pred = model.predict(X_test)
                                    accuracy = accuracy_score(y_test, y_pred)
                                    
                                    # Mostrar métricas
                                    st.write("### Desempenho do Modelo")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Acurácia", f"{accuracy:.2%}")
                                    
                                    with col2:
                                        st.metric("Classes", len(le.classes_))
                                    
                                    # Relatório de classificação
                                    st.write("### Relatório Detalhado")
                                    report = classification_report(
                                        y_test, y_pred, 
                                        target_names=le.classes_, 
                                        output_dict=True
                                    )
                                    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))
                                    
                                    # Matriz de confusão
                                    st.write("### Matriz de Confusão")
                                    cm = confusion_matrix(y_test, y_pred)
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.heatmap(
                                        cm, annot=True, fmt='d', 
                                        cmap='Blues',
                                        xticklabels=le.classes_,
                                        yticklabels=le.classes_,
                                        ax=ax
                                    )
                                    ax.set_xlabel('Predito')
                                    ax.set_ylabel('Real')
                                    ax.set_title('Matriz de Confusão')
                                    st.pyplot(fig)
                                    plt.close(fig)
                                    
                                    # Importância das features
                                    st.write("### Importância das Variáveis")
                                    importance = pd.DataFrame({
                                        'Variável': numeric_features,
                                        'Importância': model.feature_importances_
                                    }).sort_values('Importância', ascending=False)
                                    
                                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                                    sns.barplot(
                                        x='Importância', 
                                        y='Variável', 
                                        data=importance.head(10),  # Mostrar apenas as top 10
                                        palette='viridis',
                                        ax=ax2
                                    )
                                    ax2.set_title('Top 10 Variáveis Mais Importantes')
                                    st.pyplot(fig2)
                                    plt.close(fig2)
                                    
                                else:
                                    st.warning(f"Dados insuficientes após limpeza (apenas {len(df_clf)} observações válidas).")
                            else:
                                st.warning("Nenhuma feature numérica adequada encontrada para o modelo.")
                                
                        except Exception as e:
                            st.error(f"Erro no Random Forest: {str(e)}")
                    else:
                        st.warning("Nenhuma variável alvo válida selecionada para classificação.")
        
        # --- ABA 10: Discussão e Conclusão ---
        with tab10:
            st.header("Discussão e Conclusão")
            st.markdown("""
             
Os dados apresentados no gráfico de distribuição por faixa etária contém uma amostra com predomínio de pessoas com Síndrome de Down (SD) mais jovens, as categorias 5 meses a 10 anos possui 44,1% da distribuição e em segundo lugar a categoria 11 a 19 anos com 28,9% dessa demografia. Essa distribuição de uma faixa etária mais jovem é devido à natureza da pesquisa que foi aplicada em Centros de atendimento à Pessoas com Síndrome de Down, Centros Paralímpicos de atletas com SD e instituições filantrópicas e ONGs que trabalham no desenvolvimento dessa população. Em relação as características étnicas raciais, a amostra reflete um predomínio de 77,4% pessoas autodeclaradas brancas e 15,5% pardas. Analisando esses dois gráficos conseguimos notar um ponto limitante da amostra, a falta de diversidade seja pela faixa etária ou pela cor/etnia, uma vez que um dos objetivos  do projeto é avaliar características sociodemográficas de pessoas com SD no Brasil. Ao olhar para o heatmap de "Aspectos Demográficos e de Saúde" podemos obersavar uma correlação negativa mesmo que fraca entre renda familiar e cor parda (-0.21) e renda familiar e cor preta (0.16). Esses achados de renda familir verus cor/etnia refletem as desiguldades socioeconômicas históricas e estruturais no Brasil, onde populações pretas e pardas enfretam maior risco a vulnerabilidade social e econômica.

O gráfico sobre o número de irmãos que o participante possui aponta que 43,1% possui pelo menos 1 irmão. Trazendo dados da literatura, um estudo realizado na Flórida (Marshall et al., 2019), relata por parte dos pais a falta de ter recebido informações adequadas após o diagnóstico de um filho com Síndrome de Down, seja durante o período pré-natal ou no pós-natal. Em se tratando de países subdesenvolvidos essa problemática pode ser ainda pior e com uma realidade e perspectiva mais dura. Não é difícil encontrar uma sobrecarga dos cuidados recaída sobre a mãe, que na maioria das vezes precisa administrar os cuidados entre os filhos e outras demandas familiares. Ao aplicar a pesquisa foi observado, em sua grande maioria, crianças com SD sendo levadas para as atividades diárias por suas mães que em seus relatos apontavam a dificuldade em conseguir conciliar as obrigações do dia a dia.

A distribuição da renda familiar parece assimétrica para as faixas de renda mais altas, sendo 40,1% dos participantes terem uma renda de mais de 5 salários mínimos. Isso é reforçado no heatmap de "Variáveis Socioeconômicas" em que há uma correlação negativa moderada (-0.53) entre renda familiar e receber o Benefício e Pretação Continuada (BPC). Esse dado implica numa menor representação em rendas médias-baixas. Além disso, 4,6% dos participantes disseram receber o Bolsa Família e 16,3% recebem o BPC. Ainda no mesmo heatmap houve uma correlação positiva fraca (0.36) entre receber o Bolsa Família e receber o BPC. A distribuição do tipo de moradia demonstrou que 68,8% dos participantes possuem moradia própria e 19,8% possuem moradia alugada. A realidade brasileira ainda é vista por sua alta concentração de renda, onde apenas 1% da população detém de 28,3% da renda total o que reflete sobre as desigualdades sociais (IPEA, 2023). Em se tratando de pessoas com Síndrome de Down e suas família existe uma subnotificação não apenas dos aspectos sociodemográficos, mas em tantos outros que investigamos nessa pesquisa. A realidade da grande maioria dessas famílias não necessariamente está refletida nos 40,1% computados, mas sim nos relatos de pais em dizer que depender dos auxílios era algo comum, principalmente quando a pesquisa era aplicada em instituições que atendiam pessoas de baixa renda. Algumas mães relataram a necessidade de abdicar do emprego para cuidar de seus filhos de forma integral e que o auxílio era a única renda familiar.

Em relação ao nível de escolaridade dos participantes houve um predomínio de baixos níveis de escolaridade. 34,1% disseram ter Ensino Fundamental incompleto e 14,4% não se alfabetizaram e/ou não frequentaram a escola. Ainda que a amostra contenha um predomínio de pessoas na faixa etária de 5 meses a 10 anos (44,1%), 52,8% não são alfabetizados e 26,8% são semi alfabetizados, isso mesmo entre as faixas etárias adultas. Embora no heatmap de "Variáveis Socioeconômicas" exista uma correlação positiva moderada (0.56) entre idade do participante e se o participante é alfabetizado e o heatamp de "Escolaridade e Alfabetização" apresente fortes correlações positivas entre as variáveis é preciso levantar um questionamento. A realidade escolar e de qualificação profissional entre pessoas com SD e outras deficiências intelectuais é de enfrentamento de barreiras significativas, seja no acesso à informação, emprego, autonomia e participação social. Isso é reflexo tanto da falta de acessibilidade e inclusão nos mais diversos âmbitos da vida,  na falta de suporte e políticas públicas que garantem acesso à educação de qualidade, o despreparo de profissionais para atender esse público e também a superproteção e desinformação por parte do cuidador. O direito a educação inclusiva e equitativa deve ser estabelecido nas instituições. Pessoas com Síndrome de Down possuem necessidades educacionais variadas, o apoio adequado, o sentimento de pertencimento à comunidade e o suporte de autonomia garantida são alavancas que podem fazer o indivíduo com SD prosperar não apenas na vida acadêmica, mas em outros aspectos da vida (Boundy et al., 2023). Ainda sobre escolaridade e alfabetização, 64,6% dos participantes não sabe ler e 62,4% não sabe escrever e 93,3% não consegue interpretar texto. O baixo letramento funcional reflete o acesso limitado à educação e isso implicando em uma vulnerabilidade educacional exigindo planejamento e estratégias políticas que atendam as necessidade específicas dessa população.
Quanto ao nível de escolaridade do responsável do participante 33,8% disseram ter Pós-graduação e 27,5%  tem Ensino Superior completo. Esse dado reafirma o status socioeconômico da amostra e também conversa com outro dado que será apresentado na discussão, "Idade em que a mãe teve a gestação".

Analisando o gráfico da idade em que a mãe teve a gestação, majoritariamente a concentração está na faixa etária 36-40 anos (42,3%). Isso pode estar associado na decisão de ter filhos após uma estabilidade socioeconômica e planejamento familiar reforçando o perfil socioeconômico da amostra. Em relação ao acompanhamento pré-natal 66,9% disseram ter realizado na rede privada, dado consistente com o perfil de renda familiar e escolaridade do responsável. Esses achados são reforçados no heatmap de "Aspectos Demográficos e Saúde" ao encontrar uma correlação positiva fraca a moderada (0.41) entre pré-natal na rede privada e renda familiar. De forma geral, em relação as redes de acompanhamento (nutricionista, oftalmologista, psicólogo, clínico geral, dentista) a rede privada é a mais utilizada. As especialidades menos utilizadas são o acompanhamento com nutricionista e psicólogo e aqui é possível apontar uma problemática na saúde, pois pessoas com SD são mais suscetíveis a desenvolverem comorbidades, distúrbios cardiovasculares, colesterol alto, obesidade, distúrbio da tireoide e outras doenças (Asua et al., 2015). Além disso, esse público requer um acompanhamento psicológico por terem um comprometimento cognitivo, em aprendizagem, memória, linguagem e função cognitiva. Em nossa amostra, a prevalência de diagnóstico teve a ansiedade como predominante (12%). A realização de intervenção precoce seja pela rede privada (45,5%) ou pela pública (35,1%) reforça a necessidade de especialistas (fisioterapeutas, fonoaudiólogos, terapeuta ocupacional) capacitados para atender esse público, principalmente na rede pública, onde a demanda é grande e a espera por vaga no atendimento também.

Quanto ao diagnóstico do participante, 78,2% tiveram no pós-natal, sugerindo que embora o acompanhamento pré-natal seja realizado o diagnóstico de doenças genéticas acaba sendo tardio. Além disso, 93,7% dos participantes realizaram o exame de cariótipo. Este estudo aponta que métodos de triagem não invasiva são cruciais na estimativa do risco individual de uma gravidez cromossômica afetada tanto em mulheres jovens quanto em mulheres mais velhas. No entanto, apenas o diagnóstico invasivo pré-natal é definitivo da Síndrome de Down e aconselhamento genético apropriado (Vičić et al., 2017). A combinação de fatores como a escolha ou não de realiza o pré-natal, ultrassonografia morfológica não precisa, características físicas do bebê não tão evidentes ao nascimento e a necessidade de exames complementares contribuem para um diagnóstico definitivo no pós-natal.
Em relação ao irmão ter ou não alguma deficiência 98,1% disseram não ter deficiência, podendo indicar que a SD não é amplamente recorrente nas famílias da amostra.

Sobre a percepção do cuidador em relação à saúde em geral do participante, 50,7% disseram ser boa e 37,6% disseram ser excelente. Ao aplicar a pesquisa, alguns cuidadores relataram ter uma percepção positiva da saúde geral de seu filho mesmo considerando a condição genética e as comorbidades associadas, e muitas vezes comparando com outras crianças que tem Síndrome de Down ou que fazem uso regular de medicações.

Em relação ao IMC, a classificação foi realizada de acordo com a Organização Mundial da Saúde e da Sociedade Brasileira de Pediatria quando a faixa etária era de 0 meses a 10 anos devido a curva do Z escore. Ao somar as porcentagens das categorias sobrepeso e obesidades, 47,1% do sexo feminino e 43,8% do sexo masculino estão nessas classificações. Esse dado implica em um problema de saúde pública ainda mais quando o dado conversa com o gráfico de acompanhamento nutricional em que 66,9% disseram não realizar.

A distribuição de imunizações/vacinas entre os participantes apresenta uma alta cobertura para aquelas vacinas do calendário infantil anual básico. A cobertura se torna baixa para vacina da varicela, HPV, dengue e principalmente da Covid-19. Aqui vale um comentário, durante a aplicação da pesquisa houve grande resistência em obter informações sobre a vacina da Covid-19, além disso foi relatado uma grande desinformação à respeito da vacina. Infelizmente alguns cuidadores acreditam que a vacina não é benéfica à saúde de seus filhos, outros apontam que ela é causadora de outra enfermidades mesmo tendo evidências científicas de que pessoas com Síndrome de Down possuem uma baixa imunidade quando comparadas a seus pares, podendo ser consideradas imunossuprimidas devido a alterações no sistema imunológico o que as torna mais susceptíveis a infecções e doenças autoimunes (Ram et al., 2011).
A baixa taxa de casos de covid grave ou que não tiveram covid-19 pode ser reflexo de maior adesão das 1ª e 2ª doses da vacina, mas também pode ser pela não detecção da doença no período da pandemia ou mesmo na omissão do caso na hora de responder o questionário.

Entre as doenças mais contraídas apesar das imunizações, a pneumonia foi a mais frequente com 33% da amostra, sendo que este dado reforça a maior suscetibilidade dessa população em contrair infecções respiratórias, seja pelas diferenças anatômica das vias aéreas ou um sistema imunológico mais comprometido. Vale ressaltar que a vacina pneumocócica conjugada foi relativamente recente introduzida no calendário básico de imunização do Brasil, no ano de 2010, antes disso ela era direcionada apenas aos idosos.
As morbidades mais prevalentes foram distúrbio da tireoide (18%), alterações visuais (17,2%) e alterações cardiovasculares/malformações cardíacas (17%). Mapear as morbidades mais prevalentes é demonstrar os desafios de saúde pública e seu acesso.

Em relação ao uso de medicações pelos participantes 41,4% fazem suplementação de vitamina D e 30% tomam algum repositor hormonal da tireoide. Pessoas com SD podem desenvolver hipotireoidismo congênito. Um estudo de Gorini et al (2024)
mostra que a concentração baixa ou anormal de T4 ao nascer pode impactar na vida fetal e neonatal, uma vez que os hormônios tireoidianos desempenham papel fundamental no desenvolvimento do cérebro e sua deficiência contribui para a deficiência intelectual em pessoas com SD o que influência no crescimento somático e desenvolvimento psicomotor.


A percepção dos cuidadores em relação ao hábitos saudáveis dos participantes, 90,7% relatam que seus filhos possuem uma alimentação saudável, no entanto,
este dado contrasta com o gráfico de alta prevalência de sobrepeso e obesidade entre os participantes discutido anteriormente. Além disso a obesidade pode ser influenciada por fatores metabólicos intrínsecos à Síndrome de Down o que reforça a necessidade de acompanhamento nutricional.

Os gráficos de atividade física e frequência semanal praticada mostram que 37,9% dos participantes não praticam e entre àqueles que praticam a maioria pratica de 2 a 3 vezes na semana. Embora 62,1% declarem praticar alguma atividade física ou esporte, pode ser que a frequência ainda não seja o ideal para todos. Manter-se ativo é importante não apenas para o controle de peso, visto os dados de IMC dessa amostra, mas também por uma questão de saúde geral  de uma população com predisposição à obesidade e outras comorbidades (Xanthopoulos et al., 2023).

O gráfico de atendimento de saúde público adequado mostrou que 70% dos cuidadores acham que seus filhos não o recebem. Esse dado é crítico considerando que a maioria dos participantes possui acesso à rede privada. A percepção negativa ao sistema público de saúde aponta fragilidades no atendimento feito pelo SUS. Por se tratar de um público em vulnerabilidade, desigualdade e dificuldade em acessibilidade, o setor público precisa traçar estratégias políticas que atendam não apenas as demandas dessa população, mas que tenha um olhar de integralidade nos aspectos de saúde, educação e sociodemográficos. O direito à saúde pública de qualidade precisa ser garantido, mais ainda o respeito e a liberdade de ter autonomia e participação numa sociedade que precisar rever esse olhar capacitista.

Em relação aos aspectos de autonomia, participação e interação social, no gráfico de autonomia, 54,5% disseram que seus filhos tem autonomia para escolher coisas para sua satisfação pessoal. Em relação a se deslocar independentemente pela cidade 91,3% disseram que não. 51,2% é parcialmente independente na realização de suas atividade em geral. 91% se relaciona com diferentes pessoas e tem amigos. 77,1% interagem socialmente frequentando diferentes lugares da cidade e bairro. 77,1% dos cuidadores tem uma percepção positiva e acreditam que seus filhos são tratados com respeito, dignidade e igualdade pelas outras pessoas. É preciso lembrar que maior parte da faixa etária dessa amostra está entre 5 meses e 10 anos, logo não seria possível se descolar sozinho pela cidade, contudo, ainda que na vida adulta, pessoas com Síndrome de Down durante o curso da vida acabam sendo pouco estimuladas a ter sua autonomia, isso gera uma dependência e superproteção do cuidador o que implica na sua socialização e desenvolvimento. Ter uma rede de apoio é importante e ter sua independência e autonomia estimuladas é essencial para desenvolvimento de habilidades seja na capacidade de tomar decisões, resolver problemas, desenvolver autoconhecimento e capacidade de comunicação e interação social. Embora a percepção do cuidador achar que seu filho é tratado com respeito e dignidade, foi relato que em alguns casos isso ocorre porque ainda são bebês ou crianças, que ao passar para a adolescência muitos sofrem preconceito e discriminação por parte de colegas nas escolas. Essa reflexão importante, pois como mencionados vivemos em uma sociedade capacitista e desigual. Achar que pelo indivíduo ter o diagnóstico de Síndrome de Down sua vida resume a uma  crença de que ele é incapaz devido às sua deficiências é uma atitude discriminatórias. Educar e ensinar as crianças de que todos são capazes de desenvolver habilidades, seja qual for, independente de suas limitações, é um exercício de cidadania. Oportunizar, incentivar e acolher as limitações é diminuir as barreiras físicas, sociais e institucionais enfrentadas diariamente pelas pessoas com Síndrome de Down.

Ao analisar o gráfico de dispersão dos resultados da análise de agrupamento usando K-means após uma redução de dimensões por PCA (PC1 + PC2 explicam 79,5% da variância total dos dados), observamos o agrupamento em 3 clusters considerando as variáveis "idade do participante", "idade do cuidador principal", "IMC", "peso do participante", "altura do participante" e "idade em que a mãe do participante teve a gestação":

Cluster 0 - Jovens adultos/adultos com IMC alto:
A idade média dos participantes é aproximadamente 27 anos. 
A idade do cuidador é aproximadamente 62 anos (pais mais velhos no grupo). 
IMC de 28,2kg/m² (sobrepeso). 
Altura de 1,54m. 
Peso de 68kg
Idade materna na gestação entre 38-39 anos
- Este cluster representa um grupo de adultos jovens com sobrepeso, sob cuidado de pessoas mais velhas. Pode ser que os indivíduos tenham um grau maior de dependência ou vulnerabilidade física divido os pais terem mais idade.

Cluster 1 - Crianças
Idade média dos participantes de aproximadamente 3,7 anos
Idade do cuidador principal de aproximadamente 42 anos
IMC de 19,88kg/m² (eutrofia para faixa etária infantil)
Altura de 94cm
Peso de 15kg
Idade materna na gestação entre 30-31 anos
- Este cluster representa um grupo de crianças que tem cuidadores adultos na faixa etária dos 40 anos. É esperado um perfil de dependência da infância e uma demanda de cuidados completos no dia a dia.

Cluster 2 - Pré-adolescentes/adolescentes
Idade média do participante de aproximadamente 13,6 anos
Idade do cuidador principal de 49 anos
IMC de 22,9kg/m² (eutrofia)
Altura de 1,40m
Peso de 43kg
Idade materna na gestação entre 35-36 anos
- Este cluster representa um grupo de adolescentes ou pré-adolescentes com IMC compatível para a idade. Os cuidadores são pais mais jovens (pais e mãe de meia idade) quando comparado com os pais do cluster 0. É possível que este seja um grupo de pessoa com SD com maior independência funcional e autonomia quando comparado ao cluster 1.

O gráfico de distribuição da prática de atividade física temos:

Cluster 0 – Jovens adultos/adultos com IMC alto
39,2% fazem atividade 2 a 3 vezes/semana
26,4% fazem 4 a 5 vezes/semana
Apenas 20% não praticam atividade
- Este grupo é o mais ativo fisicamente. Apesar do IMC mais alto (sobrepeso), eles têm uma alta taxa de prática regular, sugerindo que estão mais conscientes da saúde ou com maior autonomia.

Cluster 1 – Crianças 
65,1% não praticam atividade física
Apenas 1,6% praticam 4 a 5 vezes/semana
- Grupo altamente sedentário, o que é esperado, considerando a idade média de ~3 anos. Muitas crianças dessa idade ainda não têm uma rotina estruturada de atividades físicas ou dependem totalmente de adultos para isso.

Cluster 2 – Pré-adolescentes/adolescentes
41,6% praticam 2 a 3 vezes/semana
10,6% praticam 4 a 5 vezes/semana
25,7% não praticam
- Este grupo apresenta um padrão intermediário com uma maior autonomia que as crianças, mas menos ativos que os adultos do cluster 0. Pode refletir fatores como falta de incentivo e oportunidade, questões sociais, limitações cognitivas/motoras, falta de acessibilidade.

O gráfico de distribuição da percepção de atendimento de saúde pública por cluster tem os seguintes achados:

Cluster 0 – Jovens adultos/adultos
45,6% dos cuidadores avaliam que o atendimento pelo sistema público de saúde é adequado, este é o maior percentual entre os grupos. Ainda assim, mais da metade (54,4%) consideram o atendimento inadequado (Não). É possível que apesar do sobrepeso identificado nesse grupo, este seja o que mais reconhece acesso ou qualidade no SUS. Provavelmente são usuários mais experientes do sistema, com certa autonomia e/ou consciência de seus direitos.

Cluster 1 – Crianças 
Apenas 18,6% disseram ter acesso a atendimento de qualidade pelo sistema público de saúde, ou seja, mais de 80% dos responsáveis não consideram o atendimento adequado. É possível que haja dificuldade dos cuidadores em conseguir atendimento especializado em demandas, por exemplo, fisioterapia, fonoaudiólogo, terapeuta ocupacional, psicólogo. Além disso, o diagnóstico precoce ou terapias contínuas para crianças com deficiência pode ser uma barreira enfrentada no dia a dia.

Cluster 2 – Pré-adolescentes/adolescentes
Apenas 24,8% avaliaram positivamente o atendimento. A percepção dos cuidadores parece um pouco melhor comparado ao cluster 1, contudo a maioria ainda é negativa. É possível que haja lacunas em acompanhamento longitudinal e de integralidade seja na reabilitação ou transição para a vida adulta.


Referências:

Marshall, J., Ramakrishnan, R., Slotnick, A. L., Tanner, J. P., Salemi, J. L., & Kirby, R. S. (2019). Family-Centered Perinatal Services for Children With Down Syndrome and Their Families in Florida. Journal of obstetric, gynecologic, and neonatal nursing : JOGNN, 48(1), 78–89. https://doi.org/10.1016/j.jogn.2018.10.006

MONTFERRE, Helio. Estudos revelam impacto da redistribuição de renda no Brasil. Ipea, 4 ago. 2023. Disponível em: https://www.ipea.gov.br/portal/categorias/45-todas-as-noticias/noticias/13909-estudos-revelam-impacto-da-redistribuicao-de-renda-no-brasil. Acesso em: 24 jun. 2025.

Boundy, L., Hargreaves, S., Baxter, R., Holton, S., & Burgoyne, K. (2023). Views of educators working with pupils with Down syndrome on their roles and responsibilities and factors related to successful inclusion. Research in developmental disabilities, 142, 104617. https://doi.org/10.1016/j.ridd.2023.104617

Vičić, A., Hafner, T., Bekavac Vlatković, I., Korać, P., Habek, D., & Stipoljev, F. (2017). Prenatal diagnosis of Down syndrome: A 13-year retrospective study. Taiwanese journal of obstetrics & gynecology, 56(6), 731–735. https://doi.org/10.1016/j.tjog.2017.10.004

Ram, G., & Chinen, J. (2011). Infections and immunodeficiency in Down syndrome. Clinical and experimental immunology, 164(1), 9–16. https://doi.org/10.1111/j.1365-2249.2011.04335.x

Gorini, F., Coi, A., Pierini, A., Assanta, N., Bottoni, A., & Santoro, M. (2024). Hypothyroidism in Patients with Down Syndrome: Prevalence and Association with Congenital Heart Defects. Children (Basel, Switzerland), 11(5), 513. https://doi.org/10.3390/children11050513

Xanthopoulos, M. S., Walega, R., Xiao, R., Pipan, M. E., Cochrane, C. I., Zemel, B. S., Kelly, A., & Magge, S. N. (2023). Physical Activity in Youth with Down Syndrome and Its Relationship with Adiposity. Journal of developmental and behavioral pediatrics : JDBP, 44(6), e436–e443. https://doi.org/10.1097/DBP.0000000000001192

            """, unsafe_allow_html=True)

            
            st.markdown("""
            <div class="justified-text">
            <h3>Principais Achados</h3>
            <p>Os resultados deste estudo revelaram padrões importantes nas características 
            sociais, educacionais e de saúde da população com síndrome de Down no Brasil:</p>
            <ul>
                <li><strong>Características Sociodemográficas</strong>: A maioria dos participantes se concentra na faixa etária X, com predominância de Y no gênero.</li>
                <li><strong>Acesso a Benefícios</strong>: Z% dos participantes recebem BPC, indicando a importância desse benefício para a população estudada.</li>
                <li><strong>Saúde</strong>: As morbidades mais prevalentes foram A e B, com C% dos participantes fazendo uso regular de medicações.</li>
                <li><strong>Educação</strong>: D% dos participantes são alfabetizados, com E% frequentando escolas regulares.</li>
            </ul>
            
            <h3>Implicações</h3>
            <p>Estes achados têm importantes implicações para políticas públicas e práticas 
            clínicas:</p>
            <ul>
                <li><strong>Políticas de Saúde</strong>: Necessidade de ampliar o acesso a especialidades médicas como F e G.</li>
                <li><strong>Educação Inclusiva</strong>: Importância de fortalecer programas de inclusão escolar e apoio pedagógico.</li>
                <li><strong>Assistência Social</strong>: Reforçar a divulgação e acesso a benefícios como o BPC e outros programas sociais.</li>
            </ul>
            
            <h3>Limitações</h3>
            <p>Algumas limitações do estudo devem ser consideradas:</p>
            <ul>
                <li><strong>Amostra</strong>: Os participantes foram recrutados principalmente através de H, o que pode limitar a generalização.</li>
                <li><strong>Dados Autorrelatados</strong>: Algumas informações dependem da percepção dos cuidadores, podendo haver viés.</li>
                <li><strong>Variáveis Não Incluídas</strong>: Fatores como I e J não foram avaliados neste estudo.</li>
            </ul>
            
            <h3>Sugestões para Pesquisas Futuras</h3>
            <p>Estudos futuros poderiam:</p>
            <ul>
                <li>Investigar a relação entre K e L utilizando métodos longitudinais.</li>
                <li>Incluir medidas objetivas de M para complementar os dados autorrelatados.</li>
                <li>Explorar as diferenças regionais no acesso a serviços e benefícios.</li>
            </ul>
            
            <h3>Conclusão</h3>
            <p>Este estudo fornece um panorama abrangente das características da população com 
            síndrome de Down no Brasil, destacando áreas prioritárias para intervenção e 
            pesquisa futura. Os resultados reforçam a necessidade de políticas intersetoriais 
            que abordem as múltiplas dimensões da vida desses indivíduos.</p>
            </div>
            """, unsafe_allow_html=True)

# --- Chamada Principal para Rodar o Dashboard ---
# Certifique-se de que a variável caminho_do_arquivo está definida corretamente acima
# Exemplo: caminho_do_arquivo = 'C:/Users/Emille/Documents/UNIFESP/MATÉRIAS/Tópicos em Ciência de Dados para Neurociência/Projeto5/Banco_SD.xlsx'

# Chamada CORRETA da função load_data:
df_original = df_original = load_data()
create_dashboard(df_original)