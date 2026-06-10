import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pokémon Analytics - Fase 2", layout="wide")

@st.cache_data
def carregar_e_processar_dados():
    df = pd.read_csv("pokemon_data.csv") 
    atributos_combate = ['HP', 'Ataque', 'Defesa', 'Sp_Atk', 'Sp_Def', 'Speed']
    
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(df[atributos_combate])
    
    K_IDEAL = 4
    kmeans = KMeans(n_clusters=K_IDEAL, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(dados_normalizados)
    score_silhueta = silhouette_score(dados_normalizados, kmeans.labels_)
    
    df['Tier'] = df['Cluster'].map({
        0: 'Standard / Intermediário',
        1: 'Underused / Abaixo da Média',
        2: 'Uber / Lendários (Outliers de Alto Poder)',
        3: 'Little Cup / Base Inicial'
    })
    
    return df, score_silhueta, K_IDEAL

try:
    df_original, score_silhueta, K_IDEAL = carregar_e_processar_dados()
except FileNotFoundError:
    st.error("Erro: O arquivo 'pokemon_data.csv' não foi encontrado na pasta. Rode o script de scraping primeiro.")
    st.stop()

atributos_combate = ['HP', 'Ataque', 'Defesa', 'Sp_Atk', 'Sp_Def', 'Speed']

st.title("Pokémon Data Mining & Analytics — Fase 2")
st.markdown("Análise estatística multidimensional, agrupamento via IA e Comparação de Criaturas.")

with st.expander("🔍 Painel de Filtros Avançados", expanded=True):
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        todos_tipos = sorted(df_original['Tipo'].unique())
        tipos_selecionados = st.multiselect(
            "Filtrar por Tipos Elementares:", 
            todos_tipos, 
            default=todos_tipos,
            help="Selecione um ou mais tipos. Deixe vazio para alertar o painel."
        )
        
    with col_f2:
        bst_min, bst_max = int(df_original['BST'].min()), int(df_original['BST'].max())
        faixa_bst = st.slider(
            "Filtrar por Total de Status (BST):", 
            bst_min, bst_max, (bst_min, bst_max)
        )

df_filtrado = df_original[
    (df_original['Tipo'].isin(tipos_selecionados)) & 
    (df_original['BST'].between(faixa_bst[0], faixa_bst[1]))
]

if df_filtrado.empty:
    st.warning("⚠️ Nenhum Pokémon encontrado com os filtros selecionados. Ajuste as opções no painel acima.")
else:
    st.markdown("### 📊 Visão Geral da População")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros Filtrados", len(df_filtrado))
    with col2:
        st.metric("Média de BST", f"{df_filtrado['BST'].mean():.1f}")
    with col3:
        st.metric("Silhouette Score (IA)", f"{score_silhueta:.3f}")
    with col4:
        st.metric("Clusters Identificados", K_IDEAL)

    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        st.subheader("Análise de Correlação: Ataque vs Defesa")
        fig_disp = px.scatter(
            df_filtrado, x='Ataque', y='Defesa', color='Tipo',
            hover_data=['Nome', 'BST'],
            labels={'Ataque': 'Poder de Ataque Físico', 'Defesa': 'Poder de Defesa Física'}
        )
        st.plotly_chart(fig_disp, use_container_width=True)

    with col_graf2:
        st.subheader("Distribuição Populacional de Poder (BST)")
        fig_hist = px.histogram(df_filtrado, x='BST', nbins=30)
        q25, q50, q75 = df_filtrado['BST'].quantile(0.25), df_filtrado['BST'].quantile(0.50), df_filtrado['BST'].quantile(0.75)
        fig_hist.add_vline(x=q25, line_dash="dash", line_color="orange", annotation_text="Q1")
        fig_hist.add_vline(x=q50, line_dash="solid", line_color="red", annotation_text="Mediana")
        fig_hist.add_vline(x=q75, line_dash="dash", line_color="orange", annotation_text="Q3")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Média de Poder (BST) por Tipo Elementar")
    media_por_tipo = df_filtrado.groupby('Tipo')['BST'].mean().reset_index().sort_values('BST', ascending=False)
    fig_barras = px.bar(
        media_por_tipo, x='Tipo', y='BST', color='Tipo',
        labels={'BST': 'Média de BST', 'Tipo': 'Tipo Elementar'}
    )
    fig_barras.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_barras, use_container_width=True)

    st.subheader("Segmentação Estatística por Machine Learning (K-Means)")
    fig_clusters = px.scatter(
        df_filtrado, x='Nome', y='BST', color='Tier',
        hover_data=atributos_combate,
        category_orders={"Tier": [
            'Uber / Lendários (Outliers de Alto Poder)', 
            'Standard / Intermediário', 
            'Underused / Abaixo da Média', 
            'Little Cup / Base Inicial'
        ]}
    )
    fig_clusters.update_xaxes(showticklabels=False)
    st.plotly_chart(fig_clusters, use_container_width=True)

    with st.expander("📄 Visualizar Base de Dados Detalhada (Tabela)", expanded=False):
        st.dataframe(df_filtrado[['Numero', 'Nome', 'Tipo', 'BST', 'Tier'] + atributos_combate], use_container_width=True)

    st.markdown("---")

    st.markdown("### ⚔️ Confronto Direto de Atributos (X1)")
    st.write("Escolha duas criaturas da base completa para comparar detalhadamente seus status e classificações.")
    
    lista_nomes = sorted(df_original['Nome'].unique())
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        poke1 = st.selectbox("Escolha o Pokémon 1:", lista_nomes, index=lista_nomes.index("Bulbasaur") if "Bulbasaur" in lista_nomes else 0)
    with col_p2:
        poke2 = st.selectbox("Escolha o Pokémon 2:", lista_nomes, index=lista_nomes.index("Charmander") if "Charmander" in lista_nomes else min(1, len(lista_nomes)-1))
        
    dados_p1 = df_original[df_original['Nome'] == poke1].iloc[0]
    dados_p2 = df_original[df_original['Nome'] == poke2].iloc[0]
    
    c_info1, c_info2 = st.columns(2)
    with c_info1:
        st.info(f"**{poke1}** ({dados_p1['Tipo']})  \n**BST:** {dados_p1['BST']} | **Tier IA:** {dados_p1['Tier']}")
    with c_info2:
        st.success(f"**{poke2}** ({dados_p2['Tipo']})  \n**BST:** {dados_p2['BST']} | **Tier IA:** {dados_p2['Tier']}")

    valores_p1 = [dados_p1[attr] for attr in atributos_combate]
    valores_p2 = [dados_p2[attr] for attr in atributos_combate]

    col_bar, col_radar = st.columns(2)

    with col_bar:
        st.markdown("#### Comparação por Atributo")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            y=atributos_combate, x=valores_p1, name=poke1, orientation='h',
            marker_color='rgb(31, 119, 180)', text=valores_p1, textposition='auto'
        ))
        fig_comp.add_trace(go.Bar(
            y=atributos_combate, x=valores_p2, name=poke2, orientation='h',
            marker_color='rgb(44, 160, 44)', text=valores_p2, textposition='auto'
        ))
        fig_comp.update_layout(
            barmode='group',
            xaxis_title="Pontuação do Atributo",
            yaxis_title="Atributos de Combate",
            height=380,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_radar:
        st.markdown("#### Radar de Atributos")
        categorias = atributos_combate + [atributos_combate[0]]
        vals_p1_radar = valores_p1 + [valores_p1[0]]
        vals_p2_radar = valores_p2 + [valores_p2[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_p1_radar, theta=categorias, fill='toself', name=poke1,
            line_color='rgb(31, 119, 180)', fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_p2_radar, theta=categorias, fill='toself', name=poke2,
            line_color='rgb(44, 160, 44)', fillcolor='rgba(44, 160, 44, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(valores_p1), max(valores_p2)) + 10])),
            showlegend=True, height=380, margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
