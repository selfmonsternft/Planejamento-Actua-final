# aplicativo_final.py (Versão Final Corrigida v14 - Detalhes no PDF de Resumo)
# Autor: Jorge & Jorge AI
# Data da Versão: 27/06/2025

import streamlit as st
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fpdf import FPDF
from io import BytesIO

try:
    from prophet import Prophet
    PROPHET_INSTALLED = True
except ImportError:
    PROPHET_INSTALLED = False

# --- Classe para Geração de PDF (Painel Detalhado) ---
class ResumoPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14); self.cell(0, 10, 'Relatório Consolidado de Planejamento', 0, 1, 'C')
        self.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, 'C'); self.ln(5)
    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    def adicionar_tabela_planejamento(self, dados, colunas_necessidade):
        # ... (código desta função não foi alterado) ...
        if dados.empty:
            self.set_font('Arial', 'I', 10); self.cell(0, 10, 'Nenhuma necessidade de compra identificada.', 0, 1, 'C'); return
        self.set_font('Arial', 'B', 9)
        colunas_basicas = [c for c in ['Cód. Item', 'Descrição do Item'] if c in dados.columns]
        colunas_exibir = colunas_basicas + colunas_necessidade
        largura_fixa_total = 100
        num_colunas_dinamicas = len(colunas_necessidade)
        try:
            largura_necessidade = (self.w - 20 - largura_fixa_total) / num_colunas_dinamicas
        except ZeroDivisionError:
            largura_necessidade = 0
        larguras = {'Cód. Item': 30, 'Descrição do Item': 70}
        for col in colunas_exibir:
            w = larguras.get(col, largura_necessidade)
            self.cell(w, 7, col.encode('latin-1', 'replace').decode('latin-1'), 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 8)
        for _, row in dados.iterrows():
            for col in colunas_exibir:
                 w = larguras.get(col, largura_necessidade)
                 valor = row.get(col, '')
                 if pd.api.types.is_numeric_dtype(type(valor)) or (isinstance(valor, (int, float))):
                     valor_str = str(int(valor))
                 else:
                     valor_str = str(valor)
                 self.cell(w, 6, valor_str.encode('latin-1', 'replace').decode('latin-1'), 1)
            self.ln()

# --- Classe para Geração de PDF (Resumo do Planejamento) ---
class ResumoPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Resumo do Planejamento Estratégico', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, dataframe):
        # ... (código desta função não foi alterado) ...
        self.set_font('Arial', '', 9)
        col_width_index = 60
        col_width_data = (self.w - self.l_margin - self.r_margin - col_width_index) / len(dataframe.columns)
        self.set_font('Arial', 'B', 9)
        index_name = dataframe.index.name if dataframe.index.name is not None else ''
        self.cell(col_width_index, 6, index_name, 1, 0, 'L')
        for col in dataframe.columns:
            self.cell(col_width_data, 6, col, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        for index, row in dataframe.iterrows():
            self.cell(col_width_index, 6, str(index), 1, 0, 'L')
            for item in row:
                self.cell(col_width_data, 6, f"{item:,.0f}", 1, 0, 'R')
            self.ln()
        self.ln(10)

    # <<< NOVO: Método para adicionar a tabela de detalhes de compra
    def adicionar_detalhe_compras(self, dados, colunas_necessidade):
        self.ln(5)
        self.chapter_title('Detalhe de Necessidades de Compra por Item (KG)')

        if dados.empty:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, 'Nenhuma necessidade de compra identificada.', 0, 1, 'L')
            self.ln(10)
            return
            
        self.set_font('Arial', 'B', 9)
        colunas_basicas = [c for c in ['Cód. Item', 'Descrição do Item'] if c in dados.columns]
        colunas_exibir = colunas_basicas + colunas_necessidade
        
        largura_fixa_total = 100
        num_colunas_dinamicas = len(colunas_necessidade)
        try:
            largura_necessidade = (self.w - self.l_margin - self.r_margin - largura_fixa_total) / num_colunas_dinamicas
        except ZeroDivisionError:
            largura_necessidade = 0
            
        larguras = {'Cód. Item': 30, 'Descrição do Item': 70}

        for col in colunas_exibir:
            w = larguras.get(col, largura_necessidade)
            self.cell(w, 7, col.encode('latin-1', 'replace').decode('latin-1'), 1, 0, 'C')
        self.ln()
        
        self.set_font('Arial', '', 8)
        for _, row in dados.iterrows():
            for col in colunas_exibir:
                 w = larguras.get(col, largura_necessidade)
                 valor = row.get(col, '')
                 if pd.api.types.is_numeric_dtype(type(valor)) or (isinstance(valor, (int, float))):
                     valor_str = str(int(valor))
                 else:
                     valor_str = str(valor)
                 self.cell(w, 6, valor_str.encode('latin-1', 'replace').decode('latin-1'), 1)
            self.ln()

# --- Funções de Apoio ---
@st.cache_data
def carregar_dados():
    # ... (código de carregamento de dados não foi alterado) ...
    df_produtos = pd.read_csv('produtos.csv', sep=';', decimal=',', converters={'Cód. Item': str})
    df_estoque = pd.read_csv('estoque.csv', sep=';', converters={'Cód. Item': str})
    df_follow_up_nac = pd.read_csv('follow_up_nacional.csv', sep=';', converters={'Cód. Item': str})
    df_follow_up_imp = pd.read_csv('follow_up_importado.csv', sep=';', converters={'Cód. Item': str})
    df_vendas = pd.read_csv('vendas_historico.csv', sep=';', converters={'Nº do item': str})
    df_previsao = pd.read_csv('previsao_vendas.csv', sep=';', converters={'Cód. Item': str})
    
    df_produtos.rename(columns={'Descrição': 'Descrição do Item'}, inplace=True, errors='ignore')
    
    df_produtos['Espessura(mm)'] = pd.to_numeric(df_produtos['Espessura(mm)'], errors='coerce')
    df_produtos['Largura(mm)'] = pd.to_numeric(df_produtos['Largura(mm)'], errors='coerce')

    df_estoque['Estoque Atual'] = pd.to_numeric(df_estoque['Estoque'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
    df_estoque = df_estoque.groupby('Cód. Item')['Estoque Atual'].sum().reset_index()

    df_vendas.rename(columns={'Nº do item': 'Cód. Item', 'Data de lançamento': 'Data Lançamento'}, inplace=True, errors='ignore')
    df_follow_up_nac.rename(columns={'Data Prevista': 'Data de Entrega', 'Quantidade': 'Saldo Pedido'}, inplace=True, errors='ignore')
    df_follow_up_imp.rename(columns={'Data Prevista': 'Data de Entrega', 'Quantidade': 'Saldo Pedido'}, inplace=True, errors='ignore')
    df_previsao.rename(columns=lambda x: x.strip(), inplace=True)
    df_previsao.rename(columns={'Data da previsão': 'Data de Previsão', 'Quantidade prevista': 'Quantidade Prevista'}, inplace=True, errors='ignore')

    for df in [df_produtos, df_estoque, df_follow_up_nac, df_follow_up_imp, df_vendas, df_previsao]:
        if 'Cód. Item' in df.columns:
            df['Cód. Item'] = df['Cód. Item'].str.zfill(5)
            
    df_follow_up_nac['Data de Entrega'] = pd.to_datetime(df_follow_up_nac['Data de Entrega'], dayfirst=True, errors='coerce')
    df_follow_up_imp['Data de Entrega'] = pd.to_datetime(df_follow_up_imp['Data de Entrega'], dayfirst=True, errors='coerce')
    if 'Data Lançamento' in df_vendas.columns:
        df_vendas['ds'] = pd.to_datetime(df_vendas['Data Lançamento'], dayfirst=True, errors='coerce')
    
    if 'Data de Previsão' in df_previsao.columns:
        mapa_meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}
        def converter_data_previsao(data_str):
            try:
                mes_str, ano_str = str(data_str).lower().split('/')
                mes = mapa_meses[mes_str]
                ano = int("20" + ano_str)
                return datetime(ano, mes, 1)
            except:
                return pd.NaT
        df_previsao['Data de Previsão'] = df_previsao['Data de Previsão'].apply(converter_data_previsao)
    
    def clean_with_thousands(series):
        return pd.to_numeric(series.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').fillna(0)

    if 'Quantidade Prevista' in df_previsao.columns:
        df_previsao['Quantidade Prevista'] = pd.to_numeric(df_previsao['Quantidade Prevista'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        df_previsao['Quantidade Prevista'] *= 1000
    if 'Saldo Pedido' in df_follow_up_imp.columns:
        df_follow_up_imp['Saldo Pedido'] = pd.to_numeric(df_follow_up_imp['Saldo Pedido'], errors='coerce').fillna(0)
    if 'Quantidade' in df_vendas.columns:
        df_vendas['y'] = clean_with_thousands(df_vendas['Quantidade'])
    if 'Saldo Pedido' in df_follow_up_nac.columns:
        df_follow_up_nac['Saldo Pedido'] = clean_with_thousands(df_follow_up_nac['Saldo Pedido'])
        
    df_vendas.dropna(subset=['ds'], inplace=True)
    df_previsao.dropna(subset=['Data de Previsão', 'Cód. Item'], inplace=True)
    df_follow_up_nac.dropna(subset=['Cód. Item', 'Data de Entrega'], inplace=True)
    df_follow_up_imp.dropna(subset=['Cód. Item', 'Data de Entrega'], inplace=True)
    df_produtos.dropna(subset=['Cód. Item', 'Descrição do Item', 'Espessura(mm)', 'Largura(mm)'], inplace=True)

    return df_produtos, df_estoque, df_follow_up_nac, df_follow_up_imp, df_vendas, df_previsao

def gerar_pdf(dataframe_necessidades, colunas_necessidade):
    pdf = RelatorioPDF(); pdf.add_page(orientation='L')
    pdf.adicionar_tabela_planejamento(dataframe_necessidades, colunas_necessidade)
    return BytesIO(pdf.output(dest='S'))

# <<< ALTERADO: Função agora recebe o df_painel para os detalhes
def gerar_resumo_pdf(df_resumo, estoque_inicial, df_painel, colunas_necessidade):
    pdf = ResumoPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Estoque Inicial Total (Hoje): {estoque_inicial:,.0f} KG", 0, 1, 'L')
    pdf.ln(5)

    df_estoque_proj = df_resumo[['Mês', 'Estoque Projetado no Fim do Mês']].set_index('Mês').T
    df_estoque_proj.index.name = 'Métricas'
    pdf.chapter_title('Estoque Total Projetado (KG)')
    pdf.chapter_body(df_estoque_proj)

    df_movimentacoes = df_resumo.set_index('Mês').drop(columns=['Estoque Projetado no Fim do Mês'])
    df_movimentacoes.index.name = 'Mês'
    pdf.chapter_title('Resumo de Entradas e Saídas Mensais (KG)')
    pdf.chapter_body(df_movimentacoes.T)

    # <<< NOVO: Adicionando a tabela de detalhes
    df_detalhe_compras = df_painel[df_painel[colunas_necessidade].sum(axis=1) > 0]
    pdf.add_page(orientation='L') # Adiciona uma nova página em modo paisagem
    pdf.adicionar_detalhe_compras(df_detalhe_compras, colunas_necessidade)

    return BytesIO(pdf.output(dest='S'))

# ... (O restante do código principal permanece o mesmo até a criação das abas) ...
# ... (código de interface e cálculo do painel) ...
st.set_page_config(layout="wide", page_title="Planejamento de Compras")
try:
    with open("style.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Arquivo 'style.css' não encontrado. O layout pode parecer simples.")

st.title("Cockpit de Planejamento de Compras")
try:
    df_produtos, df_estoque, df_follow_up_nac, df_follow_up_imp, df_vendas, df_previsao = carregar_dados()
except FileNotFoundError as e:
    st.error(f"Erro Crítico: Arquivo não encontrado - {e.filename}. Verifique se todos os arquivos .csv estão na mesma pasta do programa.")
    st.stop()
except Exception as e:
    st.error(f"Ocorreu um erro ao processar os arquivos: {e}. Verifique se as colunas dos arquivos CSV estão de acordo com o Manual.")
    st.exception(e)
    st.stop()

st.sidebar.header("Filtros da Análise")
meses_a_analisar = st.sidebar.slider("Meses Futuros a Analisar (M0 = Mês Atual)", 0, 11, 5)
grupos_de_item = sorted(df_produtos['Grupo de Item'].dropna().unique())
grupo_selecionado = st.sidebar.multiselect("Filtrar por Grupo de Item:", options=grupos_de_item, default=grupos_de_item)

if not grupo_selecionado:
    st.warning("Por favor, selecione ao menos um Grupo de Item para exibir os dados."); st.stop()

df_painel_base = df_produtos[df_produtos['Grupo de Item'].isin(grupo_selecionado)].copy()
df_painel = pd.merge(df_painel_base, df_estoque, on='Cód. Item', how='left')
df_follow_up_total = pd.concat([df_follow_up_nac[['Cód. Item', 'Saldo Pedido']], df_follow_up_imp[['Cód. Item', 'Saldo Pedido']]]).groupby('Cód. Item')['Saldo Pedido'].sum().reset_index().rename(columns={'Saldo Pedido': 'Follow Up Total'})
df_painel = pd.merge(df_painel, df_follow_up_total, on='Cód. Item', how='left')

hoje = datetime.now()
df_previsao['Mês Análise'] = (df_previsao['Data de Previsão'].dt.year - hoje.year) * 12 + (df_previsao['Data de Previsão'].dt.month - hoje.month)
df_previsao_filtrada = df_previsao[(df_previsao['Mês Análise'] >= 0) & (df_previsao['Mês Análise'] <= meses_a_analisar)]

if not df_previsao_filtrada.empty:
    previsao_mensal = df_previsao_filtrada.pivot_table(index='Cód. Item', columns='Mês Análise', values='Quantidade Prevista', aggfunc='sum').fillna(0)
    previsao_mensal.columns = [f"Previsão M{int(col)}" for col in previsao_mensal.columns]
    df_painel = pd.merge(df_painel, previsao_mensal, on='Cód. Item', how='left')

colunas_numericas_painel = ['Estoque Atual', 'Follow Up Total', 'Estoque de Segurança'] + [f'Previsão M{i}' for i in range(meses_a_analisar + 1)]
for col in colunas_numericas_painel:
    if col not in df_painel.columns:
        df_painel[col] = 0
    else:
        df_painel[col] = pd.to_numeric(df_painel[col], errors='coerce').fillna(0)

for i in range(meses_a_analisar + 1):
    df_painel[f'Necessidade M{i}'] = 0.0

for index, row in df_painel.iterrows():
    saldo_estoque_projetado = row['Estoque Atual'] + row['Follow Up Total']
    for i in range(meses_a_analisar + 1):
        previsao_mes = row.get(f'Previsão M{i}', 0)
        saldo_estoque_projetado -= previsao_mes
        estoque_seguranca = row.get('Estoque de Segurança', 0)
        necessidade = 0.0
        if saldo_estoque_projetado < estoque_seguranca:
            necessidade = estoque_seguranca - saldo_estoque_projetado
        df_painel.loc[index, f'Necessidade M{i}'] = necessidade
        saldo_estoque_projetado += necessidade

df_follow_up_completo = pd.concat([df_follow_up_nac, df_follow_up_imp])
df_follow_up_completo['Mês Análise'] = (df_follow_up_completo['Data de Entrega'].dt.year - hoje.year) * 12 + (df_follow_up_completo['Data de Entrega'].dt.month - hoje.month)
entradas_mensais = df_follow_up_completo.groupby('Mês Análise')['Saldo Pedido'].sum()

resumo = []
estoque_inicial_total = df_painel['Estoque Atual'].sum()
saldo_estoque_projetado_total = estoque_inicial_total

for i in range(meses_a_analisar + 1):
    mes_label = f"M{i}"
    vendas_previstas_mes = df_painel[f'Previsão {mes_label}'].sum()
    entradas_mes = entradas_mensais.get(i, 0)
    necessidades_mes = df_painel[f'Necessidade {mes_label}'].sum()
    
    saldo_estoque_projetado_total += entradas_mes + necessidades_mes - vendas_previstas_mes
    
    resumo.append({
        'Mês': mes_label,
        'Estoque Projetado no Fim do Mês': saldo_estoque_projetado_total,
        'Total Vendas Previstas': vendas_previstas_mes,
        'Total Entradas Programadas': entradas_mes,
        'Total Compras Sugeridas': necessidades_mes
    })

df_resumo = pd.DataFrame(resumo)

tab1, tab2, tab3, tab4 = st.tabs(["Painel de Compras", "Resumo do Planejamento", "Análise de Tendência", "Follow Up Detalhado"])

with tab1:
    st.header("Painel Principal de Necessidades")
    colunas_base = ['Cód. Item', 'Descrição do Item', 'Estoque de Segurança', 'Estoque Atual', 'Follow Up Total']
    colunas_previsao = [f'Previsão M{i}' for i in range(meses_a_analisar + 1)]
    colunas_necessidade = [f'Necessidade M{i}' for i in range(meses_a_analisar + 1)]
    colunas_finais = [col for col in colunas_base + colunas_previsao + colunas_necessidade if col in df_painel.columns]
    df_visualizacao = df_painel[colunas_finais].copy()

    colunas_de_atividade = ['Estoque de Segurança', 'Estoque Atual', 'Follow Up Total'] + colunas_previsao + colunas_necessidade
    colunas_de_atividade_existentes = [col for col in colunas_de_atividade if col in df_visualizacao.columns]

    df_visualizacao = df_visualizacao[df_visualizacao[colunas_de_atividade_existentes].sum(axis=1) != 0].reset_index(drop=True)
    
    numeric_cols = df_visualizacao.select_dtypes(include=['number']).columns
    df_visualizacao[numeric_cols] = df_visualizacao[numeric_cols].astype(int)
    
    st.dataframe(df_visualizacao)
    
    df_necessidades_pdf = df_painel[df_painel[colunas_necessidade].sum(axis=1) > 0]
    if not df_necessidades_pdf.empty:
        pdf_bytes = gerar_pdf(df_necessidades_pdf, colunas_necessidade)
        st.download_button("Baixar Relatório de Compras (PDF)", pdf_bytes, f"Relatorio_Compras_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")

with tab2:
    st.header("Resumo do Planejamento (em KG)")

    st.subheader("Estoque Total Projetado")
    st.info(f"Estoque Inicial Total (Hoje): **{estoque_inicial_total:,.0f} KG**")
    
    df_estoque_projetado = df_resumo[['Mês', 'Estoque Projetado no Fim do Mês']].set_index('Mês').T
    df_estoque_projetado_formatado = df_estoque_projetado.applymap('{:,.0f}'.format)
    st.dataframe(df_estoque_projetado_formatado)

    st.subheader("Resumo de Entradas e Saídas Mensais")
    df_resumo_formatado = df_resumo.set_index('Mês')
    df_resumo_formatado = df_resumo_formatado.drop(columns=['Estoque Projetado no Fim do Mês'])
    df_resumo_formatado_str = df_resumo_formatado.applymap('{:,.0f}'.format)
    st.dataframe(df_resumo_formatado_str)
    
    st.bar_chart(df_resumo.set_index('Mês')[['Total Vendas Previstas', 'Total Entradas Programadas', 'Total Compras Sugeridas']])
    
    st.markdown("---")
    # <<< ALTERADO: Passando os dataframes necessários para a função
    colunas_necessidade_resumo = [f'Necessidade M{i}' for i in range(meses_a_analisar + 1)]
    pdf_resumo_bytes = gerar_resumo_pdf(df_resumo, estoque_inicial_total, df_painel, colunas_necessidade_resumo)
    st.download_button(
        label="Baixar Resumo do Planejamento (PDF)",
        data=pdf_resumo_bytes,
        file_name=f"Resumo_Planejamento_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

with tab3:
    st.header("Análise Histórica e Tendência de Vendas (por Grupo)")
    if PROPHET_INSTALLED:
        if 'ds' in df_vendas.columns and 'y' in df_vendas.columns:
            itens_disponiveis = df_painel['Cód. Item'].unique()
            if len(itens_disponiveis) > 0:
                item_analise_hist = st.selectbox("Selecione um item para analisar seu grupo:", options=itens_disponiveis)
                if item_analise_hist:
                    try:
                        item_info = df_produtos.loc[df_produtos['Cód. Item'] == item_analise_hist].iloc[0]
                        CHAVE_DE_AGRUPAMENTO = ['Espessura(mm)', 'Largura(mm)', 'Grupo de Item']
                        
                        if all(col in df_produtos.columns for col in CHAVE_DE_AGRUPAMENTO):
                            df_item_selecionado = pd.DataFrame([item_info[CHAVE_DE_AGRUPAMENTO]])
                            df_item_selecionado = df_item_selecionado.astype(df_produtos[CHAVE_DE_AGRUPAMENTO].dtypes)
                            
                            itens_do_grupo = pd.merge(df_produtos, df_item_selecionado, on=CHAVE_DE_AGRUPAMENTO)
                            df_grupo_historico = df_vendas[df_vendas['Cód. Item'].isin(itens_do_grupo['Cód. Item'])].groupby('ds')['y'].sum().reset_index()
                            
                            st.write(f"Analisando grupo baseado no item {item_analise_hist}:")
                            st.write(f"Total de vendas históricas para o grupo: {df_grupo_historico['y'].sum():,.0f} KG")

                            if len(df_grupo_historico) > 12:
                                with st.spinner("Gerando gráfico de tendência..."):
                                    modelo = Prophet().fit(df_grupo_historico)
                                    futuro = modelo.make_future_dataframe(periods=180)
                                    previsao_df = modelo.predict(futuro)
                                    fig = modelo.plot(previsao_df, xlabel="Data", ylabel="Vendas Históricas e Previsão (KG)")
                                    st.pyplot(fig)
                            else:
                                st.info("Não há dados históricos suficientes (mínimo de 13 meses) para este grupo de itens.")
                        else:
                            st.warning(f"Não foi possível agrupar. Uma ou mais colunas da chave {CHAVE_DE_AGRUPAMENTO} não foram encontradas.")
                    except IndexError:
                        st.error("Item selecionado não encontrado.")
            else:
                st.info("Nenhum item disponível para análise no filtro selecionado.")
        else:
            st.warning("Colunas 'Nº do item', 'Data de lançamento' ou 'Quantidade' não encontradas ou inválidas no arquivo 'vendas_historico.csv'.")
    else:
        st.warning("A biblioteca 'Prophet' não foi encontrada. Para ativar, instale com: pip install prophet")

with tab4:
    st.header("Follow Up de Pedidos em Aberto")
    
    df_follow_nac_desc = pd.merge(df_follow_up_nac, df_produtos[['Cód. Item', 'Descrição do Item']], on='Cód. Item', how='left')
    df_follow_imp_desc = pd.merge(df_follow_up_imp, df_produtos[['Cód. Item', 'Descrição do Item']], on='Cód. Item', how='left')
    
    colunas_follow_up = ['Cód. Item', 'Descrição do Item', 'Saldo Pedido', 'Data de Entrega']
    df_follow_nac_desc = df_follow_nac_desc[colunas_follow_up]
    df_follow_imp_desc = df_follow_imp_desc[colunas_follow_up]

    st.subheader("Pedidos Nacionais")
    st.dataframe(df_follow_nac_desc)
    st.subheader("Pedidos Importados")
    st.dataframe(df_follow_imp_desc)