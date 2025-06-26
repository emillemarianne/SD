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
        #df = pd.read_excel('C:/Users/Emille/Documents/UNIFESP/MATÉRIAS/Tópicos em Ciência de Dados para Neurociência/Projeto5/Banco_SD.xlsx')
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
            st.subheader("- Distribuição por Cluster de Percepção de Atendimento de Saúde por Cluster")
            st.subheader("- Percepção de Atendimento de Saúde por Cluster")
            st.subheader("- Teste de Associação Estatística")
        
        with st.sidebar.expander("9️⃣ Modelos de Machine Learning"):
            st.header("Regressão")
            st.subheader("- Análise de Regressão")
            st.subheader("- Coeficientes do Modelo")
            st.subheader("- Diagnóstico do Modelo")
            st.header("Random Forrest")
            st.subheader("- Configuração do Modelo")
            st.subheader("- Desempenho do Modelo")
            st.subheader("- Relatório Detalhado")
            st.subheader("- Matriz de Confusão")
            st.subheader("- Importância das Variáveis")

        with st.sidebar.expander("🔟 Considerações"):
            st.header("Discussão")
            st.subheader("- Referências")
            st.subheader("- Principais Achados")
            st.subheader("- Implicações")
            st.subheader("- Limitações")
            st.subheader("- Sugestões para Pesquisas Futuras")
            
            st.header("Conclusão")
            
            
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
                    <li>O participante lê jornais, revista ou livros?</li>
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
            st.header("Análise de Correlações")
            from sklearn.preprocessing import LabelEncoder
            # Função para preprocessar dados para correlação
            def preprocess_for_correlation(df, columns):
                df_processed = df.copy()
                for col in columns:
                    if col in df_processed.columns:
                        # Tratar valores não numéricos e nulos
                        df_processed[col] = df_processed[col].astype(str).str.lower().str.strip()
                        
                        # Mapeamento para binárias 'sim'/'não'
                        if df_processed[col].nunique() <= 2 and (
                            'sim' in df_processed[col].unique() or 'não' in df_processed[col].unique() or
                            'masculino' in df_processed[col].unique() or 'feminino' in df_processed[col].unique()
                        ):
                            unique_vals = df_processed[col].unique()
                            if 'sim' in unique_vals and 'não' in unique_vals:
                                df_processed[col] = df_processed[col].map({'sim': 1, 'não': 0})
                            elif 'masculino' in unique_vals and 'feminino' in unique_vals:
                                df_processed[col] = df_processed[col].map({'masculino': 1, 'feminino': 0})
                            else: # General LabelEncoder for other binary or few categories
                                le = LabelEncoder()
                                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                        else: # Apply LabelEncoder for other categorical columns
                            le = LabelEncoder()
                            # Handle NaN values before encoding by filling with a placeholder or dropping
                            # For correlation, it's often better to drop or fill strategically.
                            # Here, fill with a string placeholder, then encode.
                            df_processed[col] = le.fit_transform(df_processed[col].fillna('N/A').astype(str))
                    else:
                        st.warning(f"Coluna '{col}' não encontrada para correlação.")
                        df_processed = df_processed.drop(columns=[col], errors='ignore') # Remove missing column
                return df_processed
        
            # --- Correlação de Variáveis Socioeconômicas ---
            st.subheader("Correlação de Variáveis Socioeconômicas")
            socioeconomic_cols = [
                'Renda familiar',
                'Recebe Bolsa família',
                'Recebe BPC',
                'A residência é',
                'Quantas pessoas vivem com esta renda familiar'
            ]
            
            # Ensure numerical conversion for 'Quantas pessoas vivem com esta renda familiar'
            if 'Quantas pessoas vivem com esta renda familiar' in df_filtrado.columns:
                df_filtrado['Quantas pessoas vivem com esta renda familiar'] = pd.to_numeric(df_filtrado['Quantas pessoas vivem com esta renda familiar'], errors='coerce')
                # Fill NaN for numerical columns before correlation calculation, e.g., with median or mean
                df_filtrado['Quantas pessoas vivem com esta renda familiar'] = df_filtrado['Quantas pessoas vivem com esta renda familiar'].fillna(df_filtrado['Quantas pessoas vivem com esta renda familiar'].median())
        
            df_socio = preprocess_for_correlation(df_filtrado, socioeconomic_cols)
            
            # Drop columns that might have been removed by preprocess_for_correlation if not found
            socioeconomic_cols_present = [col for col in socioeconomic_cols if col in df_socio.columns]
            
            if len(socioeconomic_cols_present) > 1: # Need at least 2 columns for correlation
                try:
                    corr_socio = df_socio[socioeconomic_cols_present].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_socio, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                    ax.set_title('Matriz de Correlação Socioeconômica')
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Erro ao gerar a matriz de correlação socioeconômica: {e}")
            else:
                st.warning("Não há colunas suficientes para calcular a correlação socioeconômica após o pré-processamento.")
        
            # --- Correlação de Escolaridade e Alfabetização ---
            st.subheader("Correlação de Escolaridade e Alfabetização")
            education_cols = [
                'Nível de escolaridade do participante',
                'Nível de escolaridade do responsável do participante',
                'O participante é alfabetizado',
                'Sabe ler',
                'Sabe escrever',
                'O participante consegue interpretar texto?',
                'O participante lê jornais, revistas ou livros?'
            ]
            df_edu = preprocess_for_correlation(df_filtrado, education_cols)
        
            education_cols_present = [col for col in education_cols if col in df_edu.columns]
            if len(education_cols_present) > 1:
                try:
                    corr_edu = df_edu[education_cols_present].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_edu, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                    ax.set_title('Matriz de Correlação de Escolaridade e Alfabetização')
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Erro ao gerar a matriz de correlação de escolaridade: {e}")
            else:
                st.warning("Não há colunas suficientes para calcular a correlação de escolaridade e alfabetização após o pré-processamento.")
        
            # --- Correlação de Variáveis Demográficas e de Saúde ---
            st.subheader("Correlação de Variáveis Demográficas e de Saúde")
            health_dem_cols = [
                'Idade do participante',
                'Sexo do participante',
                'Cor/etnia do participante',
                'IMC',
                'Classificação IMC', # This is categorical, will be encoded
                'Pratica alguma atividade física ou esporte',
                'Você considera que o participante possui hábitos alimentares saudáveis',
                'Você considera que o participante possui atendimento de saúde adequado pelo sistema público para atender suas necessidades',
                'Idade em que a mãe do participante teve a gestação' # Should be numeric
            ]
        
            # Ensure numerical conversion for 'Idade do participante', 'IMC', 'Idade em que a mãe do participante teve a gestação'
            for col in ['Idade do participante', 'IMC', 'Idade em que a mãe do participante teve a gestação']:
                if col in df_filtrado.columns:
                    df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')
                    df_filtrado[col] = df_filtrado[col].fillna(df_filtrado[col].median() if df_filtrado[col].dtype != 'object' else 0) # Fill NaN for numerical columns
        
            df_health_dem = preprocess_for_correlation(df_filtrado, health_dem_cols)
            
            health_dem_cols_present = [col for col in health_dem_cols if col in df_health_dem.columns]
        
            if len(health_dem_cols_present) > 1:
                try:
                    corr_health_dem = df_health_dem[health_dem_cols_present].corr()
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_health_dem, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                    ax.set_title('Matriz de Correlação de Variáveis Demográficas e de Saúde')
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Erro ao gerar a matriz de correlação de saúde e demográficas: {e}")
            else:
                st.warning("Não há colunas suficientes para calcular a correlação de variáveis demográficas e de saúde após o pré-processamento.")

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
            st.header("Discussão")
            st.markdown("""
            <div class="justified-text">
            Os dados apresentados no gráfico de distribuição por faixa etária contém uma amostra com predomínio de pessoas com Síndrome de Down (SD) mais jovens, as categorias 5 meses a 10 anos possui 44,1% da distribuição e em segundo lugar a categoria 11 a 19 anos com 28,9% dessa demografia. Essa distribuição de uma faixa etária mais jovem é devido à natureza da pesquisa que foi aplicada em Centros de atendimento à Pessoas com Síndrome de Down, Centros Paralímpicos de atletas com SD, instituições filantrópicas e ONGs que trabalham no desenvolvimento dessa população. Em relação as características étnicas raciais, a amostra reflete um predomínio de 77,4% pessoas autodeclaradas brancas e 15,5% pardas. Analisando esses dois gráficos conseguimos notar um ponto limitante da amostra, a falta de diversidade seja pela faixa etária ou pela cor/etnia, uma vez que um dos objetivos  do projeto é avaliar características sociodemográficas de pessoas com SD no Brasil. 
            
            O gráfico sobre o número de irmãos que o participante possui aponta que 43,1% possui pelo menos 1 irmão. Trazendo dados da literatura, um estudo realizado na Flórida (Marshall et al., 2019), relata por parte dos pais a falta de ter recebido informações adequadas após o diagnóstico de um filho com Síndrome de Down, seja durante o período pré-natal ou no pós-natal. Em se tratando de países subdesenvolvidos essa problemática pode ser ainda pior e com uma realidade e perspectiva mais dura. Não é difícil encontrar uma sobrecarga dos cuidados recaída sobre a mãe, que na maioria das vezes precisa administrar os cuidados entre os filhos e outras demandas familiares. Ao aplicar a pesquisa foi observado, em sua grande maioria, crianças com SD sendo levadas para as atividades diárias por suas mães que em seus relatos apontavam a dificuldade em conseguir conciliar a rotina de cuidados com o filho e as obrigações do dia a dia.
            
            A distribuição da renda familiar parece assimétrica para as faixas de renda mais altas, sendo 40,1% dos participantes terem uma renda de mais de 5 salários mínimos. Isso é reforçado no heatmap de "Correlação Socioeconômica" em que há uma correlação negativa fraca (-0.19) entre renda familiar e receber o Benefício e Pretação Continuada (BPC). Esse dado implica numa menor representação em rendas médias-baixas. Além disso, 4,6% dos participantes disseram receber o Bolsa Família e 16,3% recebem o BPC. Ainda no mesmo heatmap houve uma correlação positiva fraca (0.44) entre receber o Bolsa Família e receber o BPC. A distribuição do tipo de moradia demonstrou que 68,8% dos participantes possuem moradia própria e 19,8% possuem moradia alugada. A realidade brasileira ainda é vista por sua alta concentração de renda, onde apenas 1% da população detém de 28,3% da renda total o que reflete sobre as desigualdades sociais (IPEA, 2023). Em se tratando de pessoas com Síndrome de Down e suas família existe uma subnotificação não apenas dos aspectos sociodemográficos, mas em tantos outros que investigamos nessa pesquisa. A realidade da grande maioria dessas famílias não necessariamente está refletida nos 40,1% computados, mas sim nos relatos de pais em dizer que depender dos auxílios era algo comum, principalmente quando a pesquisa era aplicada em instituições que atendiam pessoas de baixa renda. Algumas mães relataram a necessidade de abdicar do emprego para cuidar de seus filhos de forma integral e que o auxílio era a única renda familiar.
            
            Em relação ao nível de escolaridade dos participantes houve um predomínio de baixos níveis de escolaridade. 34,1% disseram ter Ensino Fundamental incompleto e 14,4% não se alfabetizaram e/ou não frequentaram a escola. Ainda que a amostra contenha um predomínio de pessoas na faixa etária de 5 meses a 10 anos (44,1%), 52,8% não são alfabetizados e 26,8% são semi alfabetizados, isso mesmo entre as faixas etárias adultas. Embora no heatamp de "Escolaridade e Alfabetização" apresente fortes correlações positivas entre as variáveis analisadas como "saber escrever", "saber ler" é preciso levantar um questionamento. A realidade escolar e de qualificação profissional entre pessoas com SD e outras deficiências intelectuais é de enfrentamento de barreiras significativas, seja no acesso à informação, emprego, autonomia e participação social. Isso é reflexo tanto da falta de acessibilidade e inclusão nos mais diversos âmbitos da vida, na falta de suporte e políticas públicas que garantem acesso à educação de qualidade, o despreparo de profissionais para atender esse público e também a superproteção e desinformação por parte do cuidador. O direito a educação inclusiva e equitativa deve ser estabelecido nas instituições. Pessoas com Síndrome de Down possuem necessidades educacionais variadas, o apoio adequado, o sentimento de pertencimento à comunidade e o suporte de autonomia garantida são alavancas que podem fazer o indivíduo com SD prosperar não apenas na vida acadêmica, mas em outros aspectos da vida (Boundy et al., 2023). Ainda sobre escolaridade e alfabetização, 64,6% dos participantes não sabe ler e 62,4% não sabe escrever e 93,3% não consegue interpretar texto. O baixo letramento funcional reflete o acesso limitado à educação e isso implicando em uma vulnerabilidade educacional exigindo planejamento e estratégias políticas que atendam as necessidade específicas dessa população. Ao analisar o heatmap de "Escolaridade e Alfabetização" existe uma correlação negativa fraca (-0,15) entre nivel de escolaridade e ser alfabetizado, ou seja, estar em anos mais avançados na escola não é garantia que a pessoa com SD está sendo alfabetizada.
            Quanto ao nível de escolaridade do responsável do participante 33,8% disseram ter Pós-graduação e 27,5%  tem Ensino Superior completo. Esse dado reafirma o status socioeconômico da amostra e também conversa com outro dado que será apresentado na discussão, "Idade em que a mãe teve a gestação".
            
            Analisando o gráfico da idade em que a mãe teve a gestação, majoritariamente a concentração está na faixa etária 36-40 anos (42,3%). Isso pode estar associado na decisão de ter filhos após uma estabilidade socioeconômica e planejamento familiar reforçando o perfil socioeconômico da amostra. Em relação ao acompanhamento pré-natal 66,9% disseram ter realizado na rede privada, dado consistente com o perfil de renda familiar e escolaridade do responsável. De forma geral, em relação as redes de acompanhamento (nutricionista, oftalmologista, psicólogo, clínico geral, dentista) a rede privada é a mais utilizada. As especialidades menos utilizadas são o acompanhamento com nutricionista e psicólogo e aqui é possível apontar uma problemática na saúde, pois pessoas com SD são mais suscetíveis a desenvolverem comorbidades, distúrbios cardiovasculares, colesterol alto, obesidade, distúrbio da tireoide e outras doenças (Asua et al., 2015). Além disso, esse público requer um acompanhamento psicológico por terem um comprometimento cognitivo, em aprendizagem, memória, linguagem e função cognitiva. Em nossa amostra, a prevalência de diagnóstico teve a ansiedade como predominante (12%). A realização de intervenção precoce seja pela rede privada (45,5%) ou pela pública (35,1%) reforça a necessidade de especialistas (fisioterapeutas, fonoaudiólogos, terapeuta ocupacional) capacitados para atender esse público, principalmente na rede pública, onde a demanda é grande e a espera por vaga no atendimento também.
            
            Quanto ao diagnóstico do participante, 78,2% tiveram no pós-natal, sugerindo que embora o acompanhamento pré-natal seja realizado o diagnóstico de doenças genéticas acaba sendo tardio. Além disso, 93,7% dos participantes realizaram o exame de cariótipo. Este estudo aponta que métodos de triagem não invasiva são cruciais na estimativa do risco individual de uma gravidez cromossômica afetada tanto em mulheres jovens quanto em mulheres mais velhas. No entanto, apenas o diagnóstico invasivo pré-natal é definitivo da Síndrome de Down e aconselhamento genético apropriado (Vičić et al., 2017). A combinação de fatores como a escolha ou não de realiza o pré-natal, ultrassonografia morfológica não precisa, características físicas do bebê não tão evidentes ao nascimento e a necessidade de exames complementares contribuem para um diagnóstico definitivo no pós-natal.
            Em relação ao irmão ter ou não alguma deficiência 98,1% disseram não ter deficiência, podendo indicar que a SD não é amplamente recorrente nas famílias da amostra.
            
            Sobre a percepção do cuidador em relação à saúde em geral do participante, 50,7% disseram ser boa e 37,6% disseram ser excelente. Ao aplicar a pesquisa, alguns cuidadores relataram ter uma percepção positiva da saúde geral de seu filho mesmo considerando a condição genética e as comorbidades associadas, e muitas vezes comparando com outras crianças que tem Síndrome de Down ou que fazem uso regular de medicações.
            
            Em relação ao IMC, a classificação foi realizada de acordo com a Organização Mundial da Saúde e da Sociedade Brasileira de Pediatria quando a faixa etária era de 0 meses a 10 anos devido a curva do Z escore. Ao somar as porcentagens das categorias sobrepeso e obesidades, 47,1% do sexo feminino e 43,8% do sexo masculino estão nessas classificações. Esse dado implica em um problema de saúde pública ainda mais quando o dado conversa com o gráfico de acompanhamento nutricional em que 66,9% disseram não realizar.
            
            A distribuição de imunizações/vacinas entre os participantes apresenta uma alta cobertura para aquelas vacinas do calendário infantil anual básico. A cobertura se torna baixa para vacina da varicela, HPV, dengue e principalmente da Covid-19. Aqui vale um comentário, durante a aplicação da pesquisa houve grande resistência em obter informações sobre a vacina da Covid-19, além disso foi relatado uma grande desinformação à respeito da vacina. Infelizmente alguns cuidadores acreditam que a vacina não é benéfica à saúde de seus filhos, outros apontam que ela é causadora de outra enfermidades mesmo tendo evidências científicas de que pessoas com Síndrome de Down possuem uma baixa imunidade quando comparadas a seus pares, podendo ser consideradas imunossuprimidas devido a alterações no sistema imunológico o que as torna mais susceptíveis a infecções e doenças autoimunes (Ram et al., 2011).
            A baixa taxa de casos de covid grave ou que não tiveram covid-19 pode ser reflexo de maior adesão das 1ª e 2ª doses da vacina, mas também pode ser pela não detecção da doença no período da pandemia ou mesmo na omissão do caso na hora de responder o questionário.
            
            Entre as doenças mais contraídas apesar das imunizações, a pneumonia foi a mais frequente com 33% da amostra, sendo que este dado reforça a maior suscetibilidade dessa população em contrair infecções respiratórias, seja pelas diferenças anatômica das vias aéreas ou um sistema imunológico mais comprometido. Vale ressaltar que a vacina pneumocócica conjugada foi relativamente recente introduzida no calendário básico de imunização do Brasil, no ano de 2010, antes disso ela era direcionada apenas aos idosos.
            As morbidades mais prevalentes foram distúrbio da tireoide (18%), alterações visuais (17,2%) e alterações cardiovasculares/malformações cardíacas (17%). Mapear as morbidades mais prevalentes é demonstrar os desafios de saúde pública e seu acesso.
            
            Em relação ao uso de medicações pelos participantes 41,4% fazem suplementação de vitamina D e 30% tomam algum repositor hormonal da tireoide. Pessoas com SD podem desenvolver hipotireoidismo congênito. Um estudo de Gorini et al (2024) mostra que a concentração baixa ou anormal de T4 ao nascer pode impactar na vida fetal e neonatal, uma vez que os hormônios tireoidianos desempenham papel fundamental no desenvolvimento do cérebro e sua deficiência contribui para a deficiência intelectual em pessoas com SD o que influência no crescimento somático e desenvolvimento psicomotor.
            
            A percepção dos cuidadores em relação ao hábitos saudáveis dos participantes, 90,7% relatam que seus filhos possuem uma alimentação saudável, no entanto,
            este dado contrasta com o gráfico de alta prevalência de sobrepeso e obesidade entre os participantes discutido anteriormente. Além disso a obesidade pode ser influenciada por fatores metabólicos intrínsecos à Síndrome de Down o que reforça a necessidade de acompanhamento nutricional.
            
            Os gráficos de atividade física e frequência semanal praticada mostram que 37,9% dos participantes não praticam e entre àqueles que praticam a maioria pratica de 2 a 3 vezes na semana. Embora 62,1% declarem praticar alguma atividade física ou esporte, pode ser que a frequência ainda não seja o ideal para todos. Manter-se ativo é importante não apenas para o controle de peso, visto os dados de IMC dessa amostra, mas também por uma questão de saúde geral  de uma população com predisposição à obesidade e outras comorbidades (Xanthopoulos et al., 2023).
            
            O gráfico de atendimento de saúde público adequado mostrou que 70% dos cuidadores acham que seus filhos não o recebem. Esse dado é crítico considerando que a maioria dos participantes possui acesso à rede privada. A percepção negativa ao sistema público de saúde aponta fragilidades no atendimento feito pelo SUS. Por se tratar de um público em vulnerabilidade, desigualdade e dificuldade em acessibilidade, o setor público precisa traçar estratégias políticas que atendam não apenas as demandas dessa população, mas que tenha um olhar de integralidade nos aspectos de saúde, educação e sociodemográficos. O direito à saúde pública de qualidade precisa ser garantido, mais ainda o respeito e a liberdade de ter autonomia e participação numa sociedade que precisar rever esse olhar capacitista.
            
            Em relação aos aspectos de autonomia, participação e interação social, no gráfico de autonomia, 54,5% disseram que seus filhos tem autonomia para escolher coisas para sua satisfação pessoal. Em relação a se deslocar independentemente pela cidade 91,3% disseram que não. 51,2% é parcialmente independente na realização de suas atividade em geral. 91% se relaciona com diferentes pessoas e tem amigos. 77,1% interagem socialmente frequentando diferentes lugares da cidade e bairro. 77,1% dos cuidadores tem uma percepção positiva e acreditam que seus filhos são tratados com respeito, dignidade e igualdade pelas outras pessoas. É preciso lembrar que maior parte da faixa etária dessa amostra está entre 5 meses e 10 anos, logo não seria possível se descolar sozinho pela cidade, contudo, ainda que na vida adulta, pessoas com Síndrome de Down durante o curso da vida acabam sendo pouco estimuladas a ter sua autonomia, isso gera uma dependência e superproteção do cuidador o que implica na sua socialização e desenvolvimento. Ter uma rede de apoio é importante e ter sua independência e autonomia estimuladas é essencial para o desenvolvimento de habilidades seja na capacidade de tomar decisões, resolver problemas, desenvolver autoconhecimento, capacidade de comunicação e interação social. Embora a percepção do cuidador achar que seu filho é tratado com respeito e dignidade, foi relato que em alguns casos isso ocorre porque ainda são bebês ou crianças, que ao passar para a adolescência muitos sofrem preconceito e discriminação por parte de colegas nas escolas. Essa reflexão importante, pois como mencionados vivemos em uma sociedade capacitista e desigual. Achar que pelo indivíduo ter o diagnóstico de Síndrome de Down sua vida resume a uma  crença de que ele é incapaz devido às sua deficiências é uma atitude discriminatórias. Educar e ensinar as crianças de que todos são capazes de desenvolver habilidades, seja qual for, independente de suas limitações, é um exercício de cidadania. Oportunizar, incentivar e acolher as limitações é diminuir as barreiras físicas, sociais e institucionais enfrentadas diariamente pelas pessoas com Síndrome de Down.
            
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
            
            O gráfico de distribuição da percepção de atendimento de saúde pública por cluster tem os seguintes achados:
            
            Cluster 0 – Jovens adultos/adultos
            45,6% dos cuidadores avaliam que o atendimento pelo sistema público de saúde é adequado, este é o maior percentual entre os grupos. Ainda assim, mais da metade (54,4%) consideram o atendimento inadequado (Não). É possível que apesar do sobrepeso identificado nesse grupo, este seja o que mais reconhece acesso ou qualidade no SUS. Provavelmente são usuários mais experientes do sistema, com certa autonomia e/ou consciência de seus direitos.
            
            Cluster 1 – Crianças 
            Apenas 18,6% disseram ter acesso a atendimento de qualidade pelo sistema público de saúde, ou seja, mais de 80% dos responsáveis não consideram o atendimento adequado. É possível que haja dificuldade dos cuidadores em conseguir atendimento especializado em demandas, por exemplo, fisioterapia, fonoaudiólogo, terapeuta ocupacional, psicólogo. Além disso, o diagnóstico precoce ou terapias contínuas para crianças com deficiência pode ser uma barreira enfrentada no dia a dia.
            
            Cluster 2 – Pré-adolescentes/adolescentes
            Apenas 24,8% avaliaram positivamente o atendimento. A percepção dos cuidadores parece um pouco melhor comparado ao cluster 1, contudo a maioria ainda é negativa. É possível que haja lacunas em acompanhamento longitudinal e de integralidade seja na reabilitação ou transição para a vida adulta.
            
            """, unsafe_allow_html=True)

            st.header("Referências:")
            st.markdown("""

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
                <li><strong>Características Sociodemográficas</strong>: A maioria dos participantes se concentra na faixa etária de 5 meses a 10 anos.</li>
                <li><strong>Saúde</strong>: As morbidades mais prevalentes foram hipotireoidismo, doenças cardiovasculares.</li>
                <li><strong>Educação</strong>: Grande porcentagem dos participantes não são alfabetizados ou sãos semi alfabetizados.</li>
                <li><strong>Atividade física</strong>: Necessidade de aumentar a motivação e frequência na prática de ativiade física tendo em vista os altos índices de IMC.</li>
                </ul
            </div>
""", unsafe_allow_html=True)  
            

            st.markdown("""
            <div class="justified-text">
            <h3>Implicações</h3>
            <p>Estes achados têm importantes implicações para políticas públicas e práticas 
            clínicas:</p>
            <ul>
                <li><strong>Políticas de Saúde</strong>: Necessidade de ampliar o acesso a especialidades como fisioteria, fonoaudiologia, psicoligia e dente outros.</li>
                <li><strong>Educação Inclusiva</strong>: Importância de fortalecer programas de inclusão escolar, apoio pedagógico, incentivo a autonomia e participação.</li>
                <li><strong>Assistência Social</strong>: Ampliar o acesso a benefícios como o BPC e outros programas sociais.</li>
            </ul>
        </div>
""", unsafe_allow_html=True)

            st.markdown("""
            <div class="justified-text">
            <h3>Limitações</h3>
            <p>Algumas limitações do estudo devem ser consideradas:</p>
            <ul>
                <li><strong>Amostra</strong>: Os participantes foram recrutados principalmente na região Sudeste do Brasil o que pode limitar a generalização.</li>
                <li><strong>Dados Autorrelatados</strong>: Algumas informações dependem da percepção dos cuidadores, podendo haver viés.</li>   </ul>
 </div>
""", unsafe_allow_html=True)
            
            st.markdown("""
            <div class="justified-text">
            <h3>Sugestões para Pesquisas Futuras</h3>
            <p>Estudos futuros poderiam:</p>
            <ul>
                <li>Explorar as diferenças regionais no acesso a serviços e benefícios.</li>
            </ul>
        </div>
""", unsafe_allow_html=True)
            
            st.header("Conclusão")
            st.markdown("""
            <div class="justified-text">
            A análise dos dados revela importantes características sociodemográficas, educacionais, saúde e de inclusão de pessoas com Síndrome de Down no Brasil. O predomínio de crianças e adolescentes brancos com maior renda e escolaridade dos responsáveis aponta para um viés amostral, isso pode ser influenciado pelo local de aplicação da pesquisa e pela maior parte da pesquisa ter sido aplicada na região Sudeste do país. Essa limitação reduz a capacidade de generalização dos achados para toda a população com SD no país, especialmente os grupos mais vulneráveis, como aqueles em situação de pobreza extrema, de regiões periféricas ou pertencentes a minorias étnicas. Do ponto de vista socioeconômico, a distribuição de renda e o acesso majoritário à rede privada de saúde e educação reforçam esse perfil mais favorecido da amostra. No entanto, mesmo dentro desse recorte, alguns desafios como a sobrecarga materna no cuidado, o abandono do trabalho por parte dos cuidadores e a dependência de auxílios públicos em camadas economicamente mais frágeis são refletidos na discussão e literatura.
            
            A análise de escolaridade e alfabetização expõe um cenário crítico com baixos índices de letramento funcional mesmo entre participantes em idade escolar e com nível educacional regular mais avançado. Isso aponta falhas estruturais nos processos educacionais inclusivos, levantando a urgência por políticas públicas efetivas, formação adequada de profissionais e suporte às famílias.
            
            Na área da saúde, a amostra evidencia a baixa cobertura de atendimentos especializados pela rede pública, especialmente entre crianças. A prevalência de sobrepeso e comorbidades como hipotireoidismo, distúrbios cardiovasculares e ansiedade, somada à baixa adesão a acompanhamento nutricional e psicológico, reforça a necessidade de ampliação e qualificação dos serviços de saúde pública voltados para essa população. A resistência ou desinformação quanto à vacinação, sobretudo da Covid-19, também exige estratégias de educação em saúde para os cuidadores.
            
            A análise por clusters identificaram três perfis distintos (crianças, adolescentes e adultos jovens) com diferenças significativas quanto à saúde, autonomia e percepção do sistema público. Cuidadores de adultos tendem a relatar maior adequação no atendimento, possivelmente por serem usuários antigos do SUS, enquanto crianças enfrentam mais barreiras ao acesso e à continuidade dos cuidados.
            
            Por fim, os dados de autonomia e interação social sugerem avanços em aspectos afetivos e relacionais. Entretanto, apontam para um déficit na independência funcional, especialmente no deslocamento e na tomada de decisões. Isso pode ser reflexo de práticas de superproteção, falta de incentivo à autonomia e barreiras sociais ainda fortemente presentes. Embora os achados ofereçam subsídios importantes sobre a realidade de pessoas com Síndrome de Down em determinados contextos urbanos e institucionais do Brasil, a pesquisa também revela profundas desigualdades sociais e estruturais. As evidências reforçam a necessidade urgente de ações intersetoriais, políticas inclusivas, ampliação do acesso a serviços públicos de qualidade e estratégsias de conscientização que promovam equidade, autonomia e respeito aos direitos das pessoas com Síndrome de Down.
                        </div>
            """, unsafe_allow_html=True)

# --- Chamada Principal para Rodar o Dashboard ---
# Certifique-se de sque a variável caminho_do_arquivo está definida corretamente acima
# Exemplo: caminho_do_arquivo = 'C:/Users/Emille/Documents/UNIFESP/MATÉRIAS/Tópicos em Ciência de Dados para Neurociência/Projeto5/Banco_SD.xlsx'

# Chamada CORRETA da função load_data:
df_original = df_original = load_data()
create_dashboard(df_original)