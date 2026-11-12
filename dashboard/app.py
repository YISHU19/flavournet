"""FlavourNet dashboard: explore the knowledge graph built by
notebooks/flavournet_pipeline.ipynb.

    python dashboard/app.py          # http://127.0.0.1:8050
"""
from pathlib import Path

import pandas as pd
import networkx as nx
import plotly.express as px
from collections import Counter
import re, threading, time, json
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash
import dash_cytoscape as cyto

cyto.load_extra_layouts()

BASE = str(Path(__file__).resolve().parent.parent / "data") + "/"
G    = nx.read_graphml(BASE + 'flavournet_knowledge_graph.graphml')
rels = pd.read_csv(BASE + 'relationships_final.csv')
comm = pd.read_csv(BASE + 'flavournet_community_summary.csv')

# ── Node metrics ──────────────────────────────────────────
nodes_data = []
for n, d in G.nodes(data=True):
    nodes_data.append({
        'node':        n,
        'entity_type': d.get('entity_type','CONCEPT'),
        'frequency':   int(d.get('frequency',0)),
        'importance':  round(float(d.get('importance',0)),4),
        'edges':       G.degree(n),
    })
nodes_df = pd.DataFrame(nodes_data).sort_values('frequency', ascending=False).reset_index(drop=True)

edges_data = []
for u,v,d in G.edges(data=True):
    edges_data.append({
        'source':   u, 'target': v,
        'relation': d.get('relation',''),
        'weight':   float(d.get('weight',1)),
        'pmi':      float(d.get('pmi_score',0)),
    })
edges_df = pd.DataFrame(edges_data)

# ── SNA metrics ───────────────────────────────────────────
degree_cent      = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight='weight')
closeness_cent   = nx.closeness_centrality(G)
try:    pagerank = nx.pagerank(G, weight='weight')
except: pagerank = {n:0 for n in G.nodes()}

# Community lookup
node_community = {}
for _,row in comm.iterrows():
    for c in str(row['top_concepts']).split(','):
        node_community[c.strip()] = int(row['community_id'])

COLORS = {
    'DRINK':'#1D9E75','FOOD':'#f59e0b','SERVICE':'#378ADD',
    'AMBIENCE':'#a78bfa','EXPERIENCE':'#38bdf8','QUALITY':'#f87171','CONCEPT':'#888780'
}

NODE_DESC = {
    'DRINK':      'A beverage concept — drinks mentioned by customers in Coffee & Tea reviews.',
    'FOOD':       'A food item concept — dishes or ingredients mentioned in cafe reviews.',
    'SERVICE':    'A service concept — staff behaviour, speed or quality of service.',
    'AMBIENCE':   'An ambience concept — physical environment, atmosphere or setting.',
    'EXPERIENCE': 'An experience concept — when, why or how customers visit a cafe.',
    'QUALITY':    'A quality descriptor — adjectives used to evaluate products or service.',
    'CONCEPT':    'A general concept — may represent a place, person, topic or idea.',
}

REL_DESC = {
    'described_as':   'Characterised or qualified by the target concept',
    'has_quality':    'Possesses the quality or attribute of the target',
    'associated_with':'Frequently appear together in the same context',
    'relates_to':     'Grammatically or semantically linked in reviews',
    'connected_to':   'Appear in proximity within review sentences',
    'co_occurs_with': 'Appear together in the same reviews',
    'served_by':      'Served or provided by the target concept',
    'affects':        'Influences or impacts the target concept',
}

# ── Build Cytoscape elements ──────────────────────────────
def make_cy_elements(filter_type='', filter_relation='',
                     min_weight=2, min_pmi=0, search='',
                     highlight_nodes=None):
    filtered = edges_df.copy()
    if min_weight and float(min_weight) > 1:
        filtered = filtered[filtered['weight'] >= float(min_weight)]
    if min_pmi and float(min_pmi) > 0:
        filtered = filtered[filtered['pmi'] >= float(min_pmi)]
    if filter_relation:
        filtered = filtered[filtered['relation'] == filter_relation]

    visible = set(filtered['source'].tolist() + filtered['target'].tolist())

    if filter_type:
        visible = {n for n in visible if G.nodes.get(n,{}).get('entity_type','CONCEPT') == filter_type}
        filtered = filtered[filtered['source'].isin(visible) & filtered['target'].isin(visible)]

    if search and search.strip():
        s = search.lower().strip()
        matched  = {n for n in visible if s in n.lower()}
        expanded = set(matched)
        for _,row in filtered.iterrows():
            if row['source'] in matched: expanded.add(row['target'])
            if row['target'] in matched: expanded.add(row['source'])
        visible  = expanded
        filtered = filtered[filtered['source'].isin(visible) & filtered['target'].isin(visible)]

    elements = []
    for n in visible:
        if n not in G.nodes: continue
        d   = G.nodes[n]
        et  = d.get('entity_type','CONCEPT')
        fr  = int(d.get('frequency',0))
        col = COLORS.get(et,'#888780')
        sz  = 20 + min(fr/300,1)*30
        is_hl = highlight_nodes and n in highlight_nodes
        elements.append({'data':{
            'id':n,'label':n,'entity_type':et,
            'frequency':fr,'importance':round(float(d.get('importance',0)),4),
            'color':col,'size':sz,'highlighted': is_hl
        }})

    for _,row in filtered.iterrows():
        if row['source'] not in visible or row['target'] not in visible: continue
        elements.append({'data':{
            'source':row['source'],'target':row['target'],
            'relation':row['relation'],
            'weight':row['weight'],'pmi':round(row['pmi'],2),
        }})
    return elements


def make_query_elements(text, depth=2):
    tokens   = re.sub(r'[^a-z\s]',' ',text.lower()).split()
    concepts = list({t for t in tokens if len(t)>2 and t in G.nodes})
    if not concepts: return [], []

    visible       = set(concepts)
    visible_edges = []

    def expand(node_list, d):
        if d<=0: return
        nxt = []
        for n in node_list:
            if n not in G.nodes: continue
            for u,v,ed in list(G.edges(n,data=True))+list(G.in_edges(n,data=True)):
                o = v if u==n else u
                if o not in G.nodes: continue
                if o not in visible: visible.add(o); nxt.append(o)
                if (u,v) not in [(e[0],e[1]) for e in visible_edges]:
                    visible_edges.append((u,v,ed))
        expand(nxt, d-1)

    expand(concepts, depth)

    elements = []
    for n in visible:
        if n not in G.nodes: continue
        d   = G.nodes[n]
        et  = d.get('entity_type','CONCEPT')
        fr  = int(d.get('frequency',0))
        col = COLORS.get(et,'#888780')
        inp = n in concepts
        elements.append({'data':{
            'id':n,'label':n,'entity_type':et,
            'frequency':fr,'importance':round(float(d.get('importance',0)),4),
            'color':col,'size':40 if inp else 20+min(fr/300,1)*20,
            'highlighted':inp
        }})

    for u,v,ed in visible_edges:
        if u not in visible or v not in visible: continue
        elements.append({'data':{
            'source':u,'target':v,
            'relation':ed.get('relation',''),
            'weight':float(ed.get('weight',1)),
            'pmi':round(float(ed.get('pmi_score',0)),2),
        }})
    return elements, concepts


# ── Charts ────────────────────────────────────────────────
def make_freq_chart():
    top = nodes_df.head(20)
    fig = px.bar(top,x='frequency',y='node',orientation='h',
                 color='entity_type',color_discrete_map=COLORS,title='Top 20 Concepts by Frequency')
    fig.update_layout(plot_bgcolor='#111827',paper_bgcolor='#111827',font_color='#e8edf5',
                      title_font_size=13,yaxis={'categoryorder':'total ascending'},
                      height=420,margin=dict(l=10,r=10,t=40,b=10))
    return fig

def make_relation_chart():
    rc = rels['relation'].value_counts().reset_index(); rc.columns=['relation','count']
    fig = px.bar(rc,x='relation',y='count',color='count',color_continuous_scale='Teal',title='Relation Distribution')
    fig.update_layout(plot_bgcolor='#111827',paper_bgcolor='#111827',font_color='#e8edf5',
                      title_font_size=13,height=300,margin=dict(l=10,r=10,t=40,b=10),showlegend=False)
    return fig

def make_entity_chart():
    ec = nodes_df['entity_type'].value_counts().reset_index(); ec.columns=['entity_type','count']
    fig = px.pie(ec,values='count',names='entity_type',color='entity_type',color_discrete_map=COLORS,title='Entity Types')
    fig.update_layout(plot_bgcolor='#111827',paper_bgcolor='#111827',font_color='#e8edf5',
                      title_font_size=13,height=300,margin=dict(l=10,r=10,t=40,b=10))
    return fig

def make_community_chart():
    top = comm.nlargest(10,'size')
    fig = px.bar(top,x='size',y=top['community_id'].astype(str),orientation='h',
                 title='Top 10 Communities',color='size',color_continuous_scale='Viridis')
    fig.update_layout(plot_bgcolor='#111827',paper_bgcolor='#111827',font_color='#e8edf5',
                      title_font_size=13,height=320,margin=dict(l=10,r=10,t=40,b=10),showlegend=False)
    return fig

print(f"Data loaded — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

app = Dash(__name__, suppress_callback_exceptions=True)

type_opts = [{'label':'All types','value':''}]     + [{'label':t,'value':t} for t in COLORS]
rel_opts  = [{'label':'All relations','value':''}]  + [{'label':r,'value':r} for r in sorted(rels['relation'].unique())]

# ── Cytoscape stylesheet ──────────────────────────────────
CY_STYLE = [
    {'selector':'node','style':{
        'label':'data(label)',
        'background-color':'data(color)',
        'width':'data(size)','height':'data(size)',
        'font-size':'10px','color':'#e8edf5',
        'text-valign':'bottom','text-margin-y':'4px',
        'text-outline-width':'1px','text-outline-color':'#0c1117',
        'border-width':'0','opacity':'0.9',
    }},
    {'selector':'node[?highlighted]','style':{
        'border-width':'3px','border-color':'#ffffff','opacity':'1',
        'font-size':'12px','font-weight':'bold',
    }},
    {'selector':'node:selected','style':{
        'border-width':'3px','border-color':'#ffffff','opacity':'1',
    }},
    {'selector':'edge','style':{
        'line-color':'rgba(180,180,180,0.2)',
        'target-arrow-color':'rgba(180,180,180,0.3)',
        'target-arrow-shape':'triangle',
        'curve-style':'bezier',
        'width':'mapData(weight,1,500,1,6)',
        'opacity':'0.6',
    }},
    {'selector':'edge:selected','style':{
        'line-color':'#1D9E75','target-arrow-color':'#1D9E75','opacity':'1',
    }},
]

NAV  = {'background':'#0c1117','padding':'10px 20px','borderBottom':'1px solid #1e2d3d',
        'display':'flex','alignItems':'center','justifyContent':'space-between',
        'position':'sticky','top':'0','zIndex':'100'}
PAGE = {'background':'#0c1117','padding':'16px 20px','fontFamily':'DM Sans,sans-serif'}
CARD = {'background':'#111827','border':'1px solid #1e2d3d','borderRadius':'8px',
        'padding':'14px','marginBottom':'12px'}
LBL  = {'fontSize':'10px','color':'#3d5166','textTransform':'uppercase',
        'letterSpacing':'0.05em','marginBottom':'4px','display':'block'}
INP  = {'width':'100%','background':'#0c1117','border':'1px solid #1e2d3d',
        'borderRadius':'6px','padding':'7px 10px','color':'#e8edf5',
        'fontSize':'12px','fontFamily':'DM Sans,sans-serif'}
BTN  = {'width':'100%','padding':'9px','borderRadius':'6px','border':'none',
        'background':'#1D9E75','color':'#fff','fontSize':'12px',
        'fontWeight':'500','cursor':'pointer','marginTop':'8px',
        'fontFamily':'DM Sans,sans-serif'}
BTN2 = {**BTN,'background':'#1a2236','color':'#8899aa','border':'1px solid #1e2d3d'}

# ── Wider sidebar ─────────────────────────────────────────
SIDEBAR = {'width':'260px','flexShrink':'0','background':'#111827',
           'border':'1px solid #1e2d3d','borderRadius':'8px','padding':'16px',
           'overflowY':'auto','marginRight':'10px'}

# ── Taller bottom panel ───────────────────────────────────
BOTTOM = {'background':'#0a0e17','borderTop':'1px solid #1e2d3d',
          'padding':'14px 18px','height':'280px','overflowY':'auto',
          'display':'flex','gap':'0','flexShrink':'0'}

tab_s  = {'background':'transparent','border':'none','color':'#8899aa','padding':'6px 14px','fontSize':'11px'}
tab_on = {'background':'#1D9E75','border':'none','color':'#fff','padding':'6px 14px','fontSize':'11px','borderRadius':'6px'}

def stat_card(v, l, c):
    return html.Div([
        html.Div(v, style={'fontSize':'24px','fontWeight':'500','color':c,'fontFamily':'monospace'}),
        html.Div(l, style={'fontSize':'10px','color':'#8899aa','textTransform':'uppercase','marginTop':'3px'})
    ], style={**CARD,'padding':'12px 16px'})

def legend_block():
    return html.Div([
        html.Div('Legend', style={**LBL,'marginTop':'14px'}),
        html.Div([
            html.Div([
                html.Span('● ', style={'color':col,'fontSize':'14px'}),
                html.Span(et,   style={'fontSize':'11px','color':'#8899aa'})
            ], style={'marginBottom':'3px'})
            for et, col in COLORS.items()
        ])
    ])

def empty_panel(msg):
    return html.Div([
        html.Div('Node Inspector',
                 style={**LBL,'fontSize':'11px','marginBottom':'8px'}),
        html.Div(msg,
                 style={'fontSize':'12px','color':'#3d5166',
                        'marginTop':'20px','textAlign':'center','lineHeight':'1.6'})
    ], style={'flex':'1','padding':'4px 12px','borderRight':'1px solid #1e2d3d'})

def node_panel(node_id, input_concepts=None):
    if node_id not in G.nodes:
        return empty_panel(f'"{node_id}" not found')

    d   = G.nodes[node_id]
    et  = d.get('entity_type','CONCEPT')
    fr  = int(d.get('frequency',0))
    imp = float(d.get('importance',0))
    col = COLORS.get(et,'#888780')
    deg = round(degree_cent.get(node_id,0),4)
    bet = round(betweenness_cent.get(node_id,0),4)
    clo = round(closeness_cent.get(node_id,0),4)
    pgr = round(pagerank.get(node_id,0),6)
    cid = node_community.get(node_id,'—')
    is_inp = input_concepts and node_id in input_concepts
    desc   = NODE_DESC.get(et,'A concept extracted from Yelp Coffee & Tea reviews.')

    node_edges = []
    for u, v, ed in list(G.edges(node_id,data=True)) + list(G.in_edges(node_id,data=True)):
        o   = v if u == node_id else u
        dir = '→' if u == node_id else '←'
        node_edges.append((
            dir, o,
            ed.get('relation',''),
            float(ed.get('weight',1)),
            float(ed.get('pmi_score',0))
        ))
    node_edges = sorted(node_edges, key=lambda x: x[3], reverse=True)

    def mr(label, val, c='#e8edf5'):
        return html.Div([
            html.Span(label, style={'fontSize':'9px','color':'#3d5166','textTransform':'uppercase'}),
            html.Span(str(val), style={'fontSize':'11px','color':c,'fontFamily':'monospace','fontWeight':'500'}),
        ], style={'display':'flex','justifyContent':'space-between',
                  'padding':'4px 0','borderBottom':'1px solid rgba(30,45,61,0.5)'})

    # ── Column 1 — Identity & description ────────────────
    c1 = html.Div([
        html.Div([
            html.Span(node_id,
                      style={'fontSize':'16px','fontWeight':'600','color':col,'marginRight':'8px'}),
            html.Span(et,
                      style={'fontSize':'9px','padding':'2px 7px',
                             'background':col+'22','color':col,
                             'borderRadius':'3px','border':f'1px solid {col}'}),
            html.Span(' ★ Input' if is_inp else '',
                      style={'fontSize':'10px','color':'#f59e0b','marginLeft':'8px'}),
        ], style={'marginBottom':'8px'}),

        html.Div(desc,
                 style={'fontSize':'10px','color':'#8899aa','lineHeight':'1.6',
                        'background':'#0c1117','borderRadius':'6px','padding':'8px',
                        'border':'1px solid #1e2d3d','marginBottom':'8px'}),

        mr('Frequency in reviews', fr),
        mr('TF-IDF importance',    imp),
        mr('Community ID',         cid,  '#38bdf8'),
        mr('Total connections',    len(node_edges)),
    ], style={'flex':'1','minWidth':'0','paddingRight':'14px',
              'borderRight':'1px solid #1e2d3d','overflowY':'auto'})

    # ── Column 2 — SNA metrics ────────────────────────────
    c2 = html.Div([
        html.Div('SNA Metrics', style={**LBL,'marginBottom':'8px'}),
        mr('Degree centrality',      deg, '#1D9E75'),
        mr('Betweenness centrality', bet, '#378ADD'),
        mr('Closeness centrality',   clo, '#a78bfa'),
        mr('PageRank',               pgr, '#f59e0b'),
        html.Div([
            html.Div('Degree — proportion of nodes directly connected',
                     style={'fontSize':'9px','color':'#3d5166','marginTop':'8px','lineHeight':'1.4'}),
            html.Div('Betweenness — how often this bridges other concepts',
                     style={'fontSize':'9px','color':'#3d5166','marginTop':'4px','lineHeight':'1.4'}),
            html.Div('Closeness — how quickly it reaches all other nodes',
                     style={'fontSize':'9px','color':'#3d5166','marginTop':'4px','lineHeight':'1.4'}),
            html.Div('PageRank — overall influence across the network',
                     style={'fontSize':'9px','color':'#3d5166','marginTop':'4px','lineHeight':'1.4'}),
        ]),
    ], style={'width':'220px','flexShrink':'0','paddingLeft':'14px','paddingRight':'14px',
              'borderRight':'1px solid #1e2d3d','overflowY':'auto'})

    # ── Column 3 — Relationships ──────────────────────────
    def erow(dir, other, rel, w, pmi):
        oc  = COLORS.get(G.nodes.get(other,{}).get('entity_type','CONCEPT'),'#888780')
        rd  = REL_DESC.get(rel,'Related concepts')
        return html.Div([
            html.Div([
                html.Span(dir+' ', style={'color':col,'fontWeight':'600','fontSize':'12px'}),
                html.Span(other,   style={'color':oc,'fontWeight':'500','fontSize':'12px','marginRight':'6px'}),
                html.Span(f'[{rel}]', style={'color':'#3d5166','fontSize':'9px'}),
            ]),
            html.Div(
                f"{rd}  ·  weight: {int(w)}  ·  PMI: {round(pmi,2)}",
                style={'fontSize':'9px','color':'#445566','marginLeft':'14px','marginTop':'2px'}
            ),
        ], style={'padding':'5px 0','borderBottom':'1px solid rgba(30,45,61,0.3)'})

    c3 = html.Div([
        html.Div(f"Relationships ({min(len(node_edges),10)} of {len(node_edges)})",
                 style={**LBL,'marginBottom':'8px'}),
        html.Div(
            [erow(*e) for e in node_edges[:10]],
            style={'overflowY':'auto','maxHeight':'195px'}
        ),
    ], style={'flex':'1','minWidth':'0','paddingLeft':'14px','overflowY':'auto'})

    return html.Div([c1, c2, c3],
                    style={'display':'flex','height':'100%','gap':'0'})


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Span('Flavour', style={'fontSize':'18px','fontWeight':'300','color':'#e8edf5'}),
            html.Span('Net',     style={'fontSize':'18px','fontWeight':'600','color':'#1D9E75'}),
            html.Span(' · SNA Dashboard · Coffee & Tea · 10,000 reviews',
                      style={'fontSize':'10px','color':'#3d5166',
                             'fontFamily':'monospace','marginLeft':'10px'}),
        ]),
        dcc.Tabs(id='tabs', value='overview', style={'border':'none'}, children=[
            dcc.Tab(label='Overview',        value='overview',  style=tab_s, selected_style=tab_on),
            dcc.Tab(label='Knowledge Graph', value='graph',     style=tab_s, selected_style=tab_on),
            dcc.Tab(label='Query Engine',    value='query',     style=tab_s, selected_style=tab_on),
            dcc.Tab(label='Communities',     value='community', style=tab_s, selected_style=tab_on),
        ])
    ], style=NAV),

    dcc.Store(id='q-concepts-store', data=[]),
    html.Div(id='page', style={'background':'#0c1117','fontFamily':'DM Sans,sans-serif'})

], style={'background':'#0c1117','fontFamily':'DM Sans,sans-serif'})

print("Layout ready!")

@app.callback(Output('page','children'), Input('tabs','value'))
def render(tab):

    # ── OVERVIEW ──────────────────────────────────────────
    if tab == 'overview':
        return html.Div([
            html.Div([
                stat_card(str(G.number_of_nodes()),'Concept nodes',     '#1D9E75'),
                stat_card(str(G.number_of_edges()),'Relationships',     '#378ADD'),
                stat_card(str(len(comm)),           'Communities',       '#f59e0b'),
                stat_card('10,000',                 'Reviews processed', '#a78bfa'),
                stat_card('442K',                   'Coffee & Tea total','#f87171'),
            ],style={'display':'grid','gridTemplateColumns':'repeat(5,1fr)','gap':'10px','marginBottom':'14px'}),
            html.Div([
                html.Div([dcc.Graph(figure=make_freq_chart(),  config={'displayModeBar':False})],style=CARD),
                html.Div([
                    dcc.Graph(figure=make_relation_chart(),config={'displayModeBar':False}),
                    dcc.Graph(figure=make_entity_chart(),  config={'displayModeBar':False}),
                ],style=CARD),
            ],style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'12px','marginBottom':'12px'}),
            html.Div([
                html.Div('All concepts',style={**LBL,'marginBottom':'10px'}),
                dash_table.DataTable(
                    columns=[{'name':'Concept','id':'node'},{'name':'Type','id':'entity_type'},
                             {'name':'Frequency','id':'frequency'},{'name':'Importance','id':'importance'},
                             {'name':'Connections','id':'edges'}],
                    data=nodes_df.to_dict('records'),page_size=12,sort_action='native',filter_action='native',
                    style_table={'overflowX':'auto'},
                    style_header={'backgroundColor':'#1a2236','color':'#8899aa','fontWeight':'500',
                                  'fontSize':'11px','border':'1px solid #1e2d3d','textTransform':'uppercase'},
                    style_cell={'backgroundColor':'#111827','color':'#8899aa','fontSize':'12px',
                                'border':'1px solid #1e2d3d','padding':'6px 10px'},
                    style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':'#0f1a28'}],
                )
            ],style=CARD),
        ],style=PAGE)

    # ── KNOWLEDGE GRAPH ────────────────────────────────────
    elif tab == 'graph':
        init_elements = make_cy_elements()
        return html.Div([
            html.Div([
                # Sidebar
                html.Div([
                    html.Div('Filters',style={**LBL,'fontSize':'12px','color':'#e8edf5','marginBottom':'12px'}),
                    html.Span('Search node',style=LBL),
                    dcc.Input(id='g-search',placeholder='e.g. coffee...',debounce=True,
                              style={**INP,'marginBottom':'10px'}),
                    html.Span('Entity type',style=LBL),
                    dcc.Dropdown(id='g-type',options=type_opts,value='',clearable=True,
                                 style={'marginBottom':'10px','fontSize':'12px','color':'#111'}),
                    html.Span('Relation type',style=LBL),
                    dcc.Dropdown(id='g-rel',options=rel_opts,value='',clearable=True,
                                 style={'marginBottom':'10px','fontSize':'12px','color':'#111'}),
                    html.Div([html.Span('Min weight: ',style=LBL),
                              html.Span(id='w-lbl',style={'fontSize':'11px','color':'#1D9E75','fontFamily':'monospace'})],
                             style={'display':'flex','alignItems':'center','marginBottom':'3px'}),
                    dcc.Slider(id='g-weight',min=1,max=200,step=5,value=2,marks=None,
                               tooltip={'placement':'bottom','always_visible':False}),
                    html.Div([html.Span('Min PMI: ',style=LBL),
                              html.Span(id='p-lbl',style={'fontSize':'11px','color':'#1D9E75','fontFamily':'monospace'})],
                             style={'display':'flex','alignItems':'center','marginTop':'8px','marginBottom':'3px'}),
                    dcc.Slider(id='g-pmi',min=0,max=10,step=0.5,value=0,marks=None,
                               tooltip={'placement':'bottom','always_visible':False}),
                    html.Button('Apply filters',id='g-apply',n_clicks=0,style=BTN),
                    html.Button('Reset all',    id='g-reset',n_clicks=0,style=BTN2),
                    legend_block(),
                ],style=SIDEBAR),

                # Graph + bottom panel
                html.Div([
                    cyto.Cytoscape(
                        id='g-cyto',
                        elements=init_elements,
                        style={'width':'100%','height':'calc(100vh - 86px - 210px)',
                               'background':'#111827'},
                        layout={'name':'cose','animate':True,'randomize':True,
                                'componentSpacing':80,'nodeRepulsion':800000,
                                'edgeElasticity':100,'nestingFactor':5,
                                'gravity':80,'numIter':1000,'initialTemp':200,
                                'coolingFactor':0.95,'minTemp':1.0},
                        stylesheet=CY_STYLE,
                        minZoom=0.1, maxZoom=3,
                        userZoomingEnabled=True,
                        userPanningEnabled=True,
                    ),
                    # Bottom panel
                    html.Div(
                        id='g-panel',
                        children=empty_panel('Click any node to inspect its SNA metrics, relationships and PMI scores'),
                        style=BOTTOM
                    ),
                ],style={'flex':'1','display':'flex','flexDirection':'column','overflow':'hidden',
                         'background':'#111827','borderRadius':'8px','border':'1px solid #1e2d3d'}),

            ],style={'display':'flex','height':'calc(100vh - 86px)','gap':'0',
                     'padding':'0','alignItems':'stretch'}),
        ],style={**PAGE,'padding':'16px 20px'})

    # ── QUERY ENGINE ───────────────────────────────────────
    elif tab == 'query':
        return html.Div([
            html.Div([
                # Sidebar
                html.Div([
                    html.Div('Text-to-Network Engine',
                             style={'fontSize':'14px','fontWeight':'600','color':'#e8edf5','marginBottom':'4px'}),
                    html.Div('Type any coffee shop review — concepts are extracted and their sub-graph shown',
                             style={'fontSize':'11px','color':'#8899aa','marginBottom':'12px','lineHeight':'1.5'}),
                    html.Span('Your review',style=LBL),
                    dcc.Textarea(id='q-text',
                                 value='The barista was friendly but the espresso was too bitter and the wait was long',
                                 style={**INP,'height':'100px','resize':'none','marginBottom':'10px'}),
                    html.Span('Expansion depth',style=LBL),
                    dcc.Slider(id='q-depth',min=1,max=3,step=1,value=2,
                               marks={1:'1 hop',2:'2 hops',3:'3 hops'},
                               tooltip={'placement':'bottom','always_visible':False}),
                    html.Button('Build knowledge graph',id='q-build',n_clicks=0,
                                style={**BTN,'marginTop':'12px','background':'#185FA5'}),
                    html.Button('Clear',id='q-clear',n_clicks=0,style=BTN2),
                    html.Div(id='q-info',
                             style={'fontSize':'11px','color':'#8899aa','marginTop':'12px','lineHeight':'1.7'}),
                    legend_block(),
                ],style=SIDEBAR),

                # Query graph + bottom panel
                html.Div([
                    cyto.Cytoscape(
                        id='q-cyto',
                        elements=[],
                        style={'width':'100%','height':'calc(100vh - 86px - 210px)',
                               'background':'#111827'},
                        layout={'name':'cose','animate':True,'randomize':True,
                                'nodeRepulsion':600000,'edgeElasticity':100,
                                'gravity':80,'numIter':800},
                        stylesheet=CY_STYLE,
                        minZoom=0.2, maxZoom=4,
                        userZoomingEnabled=True,
                        userPanningEnabled=True,
                    ),
                    html.Div(
                        id='q-panel',
                        children=empty_panel('Build a graph then click any node to see its description and relationships'),
                        style=BOTTOM
                    ),
                ],style={'flex':'1','display':'flex','flexDirection':'column','overflow':'hidden',
                         'background':'#111827','borderRadius':'8px','border':'1px solid #1e2d3d'}),

            ],style={'display':'flex','height':'calc(100vh - 86px)','gap':'0','alignItems':'stretch'}),
        ],style={**PAGE,'padding':'16px 20px'})

    # ── COMMUNITIES ────────────────────────────────────────
    elif tab == 'community':
        return html.Div([
            html.Div([
                stat_card(str(len(comm)),          'Communities','#1D9E75'),
                stat_card(str(comm['size'].max()), 'Largest',    '#f59e0b'),
                stat_card('Louvain',               'Algorithm',  '#a78bfa'),
            ],style={'display':'grid','gridTemplateColumns':'repeat(3,1fr)','gap':'10px','marginBottom':'14px'}),
            html.Div([
                html.Div([dcc.Graph(figure=make_community_chart(),config={'displayModeBar':False})],style=CARD),
                html.Div([
                    html.Div('Community breakdown',style={**LBL,'marginBottom':'10px'}),
                    dash_table.DataTable(
                        columns=[{'name':'ID','id':'community_id'},{'name':'Size','id':'size'},
                                 {'name':'Dominant type','id':'dominant_type'},
                                 {'name':'Top concepts','id':'top_concepts'}],
                        data=comm.to_dict('records'),page_size=12,sort_action='native',
                        style_table={'overflowX':'auto'},
                        style_header={'backgroundColor':'#1a2236','color':'#8899aa','fontWeight':'500',
                                      'fontSize':'11px','border':'1px solid #1e2d3d','textTransform':'uppercase'},
                        style_cell={'backgroundColor':'#111827','color':'#8899aa','fontSize':'12px',
                                    'border':'1px solid #1e2d3d','padding':'6px 10px',
                                    'whiteSpace':'normal','height':'auto'},
                        style_data_conditional=[{'if':{'row_index':'odd'},'backgroundColor':'#0f1a28'}],
                    )
                ],style=CARD),
            ],style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':'12px'}),
        ],style=PAGE)


# ── Knowledge graph update ────────────────────────────────
@app.callback(
    Output('g-cyto','elements'),
    Input('g-apply','n_clicks'),
    Input('g-reset','n_clicks'),
    Input('tabs','value'),
    State('g-search','value'),
    State('g-type','value'),
    State('g-rel','value'),
    State('g-weight','value'),
    State('g-pmi','value'),
    prevent_initial_call=False
)
def update_graph(ac,rc,tab,search,gtype,grel,weight,pmi):
    if tab != 'graph': return dash.no_update
    ctx     = callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    if 'g-reset' in trigger:
        return make_cy_elements()
    return make_cy_elements(
        filter_type     = gtype  or '',
        filter_relation = grel   or '',
        min_weight      = float(weight) if weight else 2,
        min_pmi         = float(pmi)    if pmi    else 0,
        search          = search or '',
    )


# ── Graph node click → bottom panel ──────────────────────
@app.callback(
    Output('g-panel','children'),
    Input('g-cyto','tapNodeData'),
    prevent_initial_call=True
)
def graph_node_click(data):
    if not data:
        return empty_panel('Click any node to inspect')
    return node_panel(data['id'], input_concepts=None)


# ── Query engine ──────────────────────────────────────────
@app.callback(
    Output('q-cyto','elements'),
    Output('q-info','children'),
    Output('q-concepts-store','data'),
    Input('q-build','n_clicks'),
    Input('q-clear','n_clicks'),
    State('q-text','value'),
    State('q-depth','value'),
    prevent_initial_call=True
)
def update_query(bc,cc,text,depth):
    ctx     = callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    if 'q-clear' in trigger:
        return [], html.Div('Cleared. Type a new review.',style={'color':'#8899aa'}), []

    if not text or not text.strip():
        return dash.no_update, html.Div('Please enter review text.',style={'color':'#f87171'}), []

    elements, concepts = make_query_elements(text.strip(), int(depth) if depth else 2)

    if not concepts:
        return dash.no_update, html.Div([
            html.Div('No known concepts found.',style={'color':'#f87171','marginBottom':'4px','fontWeight':'500'}),
            html.Div('Try: coffee, barista, latte, espresso, wait, service, food, atmosphere, friendly...',
                     style={'color':'#8899aa','lineHeight':'1.5'}),
        ]), []

    info = html.Div([
        html.Div([html.Span('Input concepts: ',style={'color':'#3d5166'}),
                  html.Span(', '.join(concepts),style={'color':'#1D9E75','fontWeight':'500'})],
                 style={'marginBottom':'5px'}),
        html.Div([html.Span('Depth: ',style={'color':'#3d5166'}),
                  html.Span(f"{depth} hop{'s' if int(depth or 2)>1 else ''}",style={'color':'#8899aa'})],
                 style={'marginBottom':'4px'}),
        html.Div('Larger nodes = input concepts · click any node for details',
                 style={'color':'#3d5166','marginTop':'4px'}),
    ])
    return elements, info, concepts


# ── Query node click → bottom panel ──────────────────────
@app.callback(
    Output('q-panel','children'),
    Input('q-cyto','tapNodeData'),
    State('q-concepts-store','data'),
    prevent_initial_call=True
)
def query_node_click(data, concepts):
    if not data:
        return empty_panel('Click any node to inspect')
    return node_panel(data['id'], input_concepts=set(concepts or []))


# ── Slider labels ──────────────────────────────────────────
@app.callback(
    Output('w-lbl','children'),
    Output('p-lbl','children'),
    Input('g-weight','value'),
    Input('g-pmi','value'),
)
def update_labels(w,p):
    return str(int(w) if w else 2), str(p if p else 0)

print("All callbacks ready!")

if __name__ == "__main__":
    app.run(port=8050, debug=False)
