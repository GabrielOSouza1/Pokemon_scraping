# 📊 Pokémon Data Mining & Analytics — Fase 2

Trabalho prático de Mineração de Dados — dashboard analítico com Machine Learning aplicado a dados de Pokémon coletados via web scraping.

## 🗂️ Estrutura do Projeto

```
pokemon/
├── dashboard.py          # Dashboard interativo (Streamlit)
├── pokemon_data.csv      # Dataset gerado pelo scraping
├── .gitignore
└── README.md
```

## ⚙️ Requisitos

- Python 3.8+
- Instalar as dependências:

```bash
pip install streamlit pandas numpy scikit-learn plotly requests beautifulsoup4
```

## ▶️ Como Executar

**1. (Opcional) Regerar o dataset:**
```bash
python scraping_pokemon.py
```
> O arquivo `pokemon_data.csv` já está incluído no repositório, esse passo não é obrigatório.

**2. Rodar o dashboard:**
```bash
streamlit run dashboard.py
```

**3. Acessar no navegador:**
```
http://localhost:8501
```
> Mantenha o terminal aberto enquanto usa o dashboard.

## 📈 Funcionalidades

- **Filtros globais** por Tipo Elementar e faixa de BST
- **Cards de indicadores** com total de registros, média de BST, Silhouette Score e número de clusters
- **Gráfico de dispersão** Ataque vs Defesa por tipo
- **Histograma** de distribuição de poder (BST) com quartis
- **Gráfico de barras** com média de BST por tipo elementar
- **Segmentação K-Means** com classificação por tiers de poder
- **Comparador X1** com gráfico de barras e radar chart entre dois Pokémon

## 🤖 Técnica de Machine Learning

Agrupamento não supervisionado via **K-Means** (K=4) com normalização dos dados via `StandardScaler`. A qualidade do modelo é avaliada pelo **Silhouette Score**.

## 📦 Fonte dos Dados

[Pokémon Database — pokemondb.net/pokedex/all](https://pokemondb.net/pokedex/all)
