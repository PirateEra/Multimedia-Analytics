import dash
import nltk
import plotly.graph_objs as go
import pandas as pd
from io import StringIO
from src.utils.seed import seed_everything
from dash import dcc, html, Input, Output, State, ctx, Dash
import diskcache
import dash_cytoscape as cyto
import plotly.express as px
from datasetloader import DatasetLoader
from dash.exceptions import PreventUpdate
from pprint import pprint
from infer_sample import (
    multiple_queries,
    jaccard_similarity,
    format_edges_for_prompt,
    get_subgraph,
    predict_graph,
    get_node_labels_from_nodes_df,
    multiple_queries_clauses,
    graph_desc_to_graph
)
# Global variables used during the app
LOADED_DATASETS = {
    "scene_graphs": DatasetLoader("scene_graphs"),
    "expla_graphs": DatasetLoader("expla_graphs"),
    "webqsp": DatasetLoader("webqsp")
}
CURRENT_DATASET = None
CURRENT_GRAPH = None
#
app = Dash(__name__)
#######
## Helper function to get all the defined styles for a graph
#######
def get_styles():
    styles = []
    #----
    # main graph
    #---
    styles.append({
        'selector': '.basic-node',
        'style': {
            'label': 'data(label)',
            'background-color': '#4a90e2',
            'width': 50,
            'height': 50,
            'font-size': 12,
            'color': '#ffffff',
            'text-valign': 'center',
            'text-halign': 'center',
            'border-width': 1,
            'border-color': '#2c3e50',
            'text-outline-color': '#4a90e2',
            'text-outline-width': 1,
        }
    })

    styles.append({
        'selector': '.basic-edge',
        'style': {
            'label': 'data(label)',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1,
            'width': 1.5,
            'line-color': '#999',
            'target-arrow-color': '#999',
            'font-size': 10,
            'color': '#555',
            'text-rotation': 'autorotate',
            'text-margin-y': -8,
            'text-background-color': '#ffffff',
            'text-background-opacity': 0.85,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle',
        }
    })

    #---
    # Subgraph
    #---
    styles.append({
        'selector': '.subgraph-node',
        'style': {
            'label': 'data(label)',
            'background-color': '#f9a825',
            'width': 75,
            'height': 75,
            'font-size': 15,
            'font-weight': 'bold',
            'color': '#212121',
            'text-valign': 'center',
            'text-halign': 'center',
            'border-width': 4,
            'border-color': '#f57f17',
            'text-outline-color': '#fff9c4',
            'text-outline-width': 2,
        }
    })

    styles.append({
        'selector': '.subgraph-edge',
        'style': {
            'label': 'data(label)',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.4,
            'width': 3,
            'line-color': '#f9a825',
            'target-arrow-color': '#f9a825',
            'font-size': 11,
            'font-weight': 'bold',
            'color': '#1a1a1a',
            'text-rotation': 'autorotate',
            'text-margin-y': -10,
            'text-background-color': '#fffde7',
            'text-background-opacity': 0.95,
            'text-background-padding': '3px',
            'text-background-shape': 'roundrectangle',
            'opacity': 1.0
        }
    })



    return styles


#######
## App layout defining (the actual element creation)
#######

app.layout = html.Div([

    #######
    ## Left sidebar
    #######
    dcc.Store(id="subgraph-selection"),
    dcc.Store(id="jaccard-graph"),
    dcc.Store(id="query-output-data"),
    dcc.Store(id="node-info-data"),
    dcc.Store(id="query-wipe-data-select", data=False),
    dcc.Store(id="query-wipe-graph-select", data=False),
    html.Div([
        #----
        # All selection dropdowns
        #----
        html.H3("Dataset"),
        dcc.Dropdown(
            id='dataset-selector',
            options=[{"label": name, "value": name} for name in LOADED_DATASETS.keys()],
            value=list(LOADED_DATASETS.keys())[0],  # pick first dataset as default
            clearable=False
        ),

        html.H3("Graph"),
        dcc.Dropdown(id='graph-selector', clearable=False),

        html.H3("Query"),
        dcc.Textarea(
            id='query-text',
            value='Mention something interesting about the current graph.',
            style={
                'width': '100%', 
                'height': 150,
                'backgroundColor': '#f9f9f9',
                'resize': 'none',
            }
        ),
        html.Div([
            html.Label("Seed:"),
            dcc.Input(
                id='seed-input',
                value=1,
                type='number',
                placeholder='Enter seed value',
                style={'width': '100%'}
            )
        ], style={'marginTop': '10px'}),
        html.Button('Run Query', id='run-query-button', disabled=False, style={'marginTop': '10px'}),

        dcc.Checklist(
            id='compute-jaccard',
            options=[{'label': 'Compute significance scores', 'value': 'enabled'}],
            value=[]
        ),

        html.H3("Visible Nodes"),
        html.Div(id='output'),
        dcc.RangeSlider(
            id='node-id-slider',
            min=0,
            max=100,
            step=1,
            value=[0, 100],
            marks={},
            tooltip={"placement": "bottom", "always_visible": False},
            allowCross=False,
            updatemode='mouseup'
        ),

        dcc.Checklist(
            id='show-self-loops',
            options=[{'label': 'Show self loops', 'value': 'enabled'}],
            value=[]
        ),
    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    #######
    ## Main content: Subgraph + Query Output side by side
    #######
    html.Div([
        html.Div([
            #----
            # Graph viewer title and checkbox
            #----
            html.Div(
                style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'},
                children=[
                    html.H3("Graph Viewer"),
                    dcc.Checklist(
                        options=[{'label': 'Show Subgraph Only', 'value': 'enabled'}],
                        value=[],
                        id='show-subgraph-only',
                        inputStyle={"margin-left": "10px"}
                    ),
                ]
            ),

            #----
            # graph visualisation
            #----
            cyto.Cytoscape(
                id='cytoscape-graph',
                layout={
                    'name': 'cose',
                    'nodeRepulsion': 8000,
                    'idealEdgeLength': 100,
                    'gravity': 0.0,
                    'animate': True
                },
                style={'width': '100%', 'height': '400px'}
            ),

            #----
            # Node info
            #----
            html.H4("Node-Info"),
            dcc.Textarea(
                id='node-info-text-area',
                value='Click a node to show more info',
                readOnly=True,
                style={
                    'width': '95%',
                    'height': '100px',
                    'resize': 'none',
                    'overflowY': 'auto',
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'backgroundColor': '#f9f9f9',
                    'whiteSpace': 'pre-wrap',
                    'fontFamily': 'inherit',
                    'fontSize': '14px',
                }
            )
        ], style={'width': '59%', 'display': 'inline-block', 'verticalAlign': 'top', 'boxSizing': 'border-box'}),



        html.Div([
            #----
            # Entire query output section (right hand side of the page)
            #----
            html.H4("Query Output"),
            html.Div(
                style={
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'minHeight': '400px',
                    'whiteSpace': 'pre-wrap',
                    'backgroundColor': '#f9f9f9',
                    'position': 'relative'
                },
                children=[
                    #----
                    # Loading spinner
                    #----
                    html.Div(
                        style={
                            'position': 'absolute',
                            'top': '50%',
                            'left': '50%',
                            'transform': 'translate(-50%, -50%)',
                            'zIndex': 1
                        },
                        children=dcc.Loading(
                            type="default",
                            children=html.Div(id='trigger-loading', style={'display': 'none'})
                        )
                    ),
                    #----
                    # Query output box
                    #----
                    html.Div(id='query-output', style={'position': 'relative', 'zIndex': 0})
                ]
            ),

            #----
            # jaccard graph, normally disable due to display none, but dynamically set to display based on input of user
            #----
            html.Div(
                id='jaccard-bar-container',
                style={
                    'marginTop': '10px',
                    'display': 'none',
                    'backgroundColor': '#f9f9f9',
                    'border': '1px solid #ccc',
                    'padding': '5px',
                    'borderRadius': '5px',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.05)'
                },
                children=[
                    html.Div("Significance Scores", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Graph(
                        id='jaccard-bar-chart',
                        config={'displayModeBar': False},
                        style={'height': '200px'},
                    )
                ]
            ),
        ], style={'width': '38%', 'display': 'inline-block', 'paddingLeft': '10px', 'verticalAlign': 'top', 'boxSizing': 'border-box'})

    ], style={'width': '70%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'})


])

#######
## Callback for updating the list of all graphs based on the dataset that is currently active
#######
@app.callback(
    Output('graph-selector', 'options'),
    Output('graph-selector', 'value'),
    Output('query-wipe-data-select', 'data'),
    Input('dataset-selector', 'value')
)
def update_graph_list(selected_dataset_name):
    global CURRENT_DATASET
    CURRENT_DATASET = LOADED_DATASETS[selected_dataset_name]

    graph_ids = CURRENT_DATASET.list_graph_ids()

    options = [{"label": str(gid), "value": str(gid)} for gid in graph_ids]
    first_value = options[0]['value'] if options else None

    return options, first_value, True

#######
## Callback for showing the current graph (center element of the page), based on the currently selected graph from the dropdown
#######
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Output('cytoscape-graph', 'stylesheet'),
    Output('node-id-slider', 'min'),
    Output('node-id-slider', 'max'),
    Output('node-id-slider', 'marks'),
    Output('node-id-slider', 'value'),
    Output('query-wipe-graph-select', 'data'),
    Input('show-self-loops', 'value'),
    Input('subgraph-selection', 'data'),
    Input('graph-selector', 'value'),
    State('dataset-selector', 'value'),
    Input('node-id-slider', 'value'),
    State('node-id-slider', 'min'),
    State('node-id-slider', 'max'),
    State('node-id-slider', 'marks'),
    State('show-self-loops', 'value'),
    Input('jaccard-bar-chart', 'selectedData'),
    State('jaccard-graph', 'data'),
    Input('show-subgraph-only', 'value'),
    State('show-subgraph-only', 'value')
)
def update_graph(_, data, selected_graph_id, selected_dataset_name, slider_value, min_value, max_value, marks, show_self_loops, jaccard_click_data, jaccard_data, subgraph_tick_input, subgraph_tick_value):
    if not selected_graph_id:
        global CURRENT_GRAPH
        selected_graph_id = CURRENT_GRAPH
    elif not selected_dataset_name:
        global CURRENT_DATASET
        selected_dataset_name = CURRENT_DATASET

    # Do the actual graph creation
    dataset_loader = LOADED_DATASETS[selected_dataset_name]
    _, nodes_df, edges_df = dataset_loader.load_graph_by_id(selected_graph_id)

    # deal with the node slider
    if ctx.triggered_id != "node-id-slider":
        min_value, max_value, marks, slider_value = update_node_slider(nodes_df)
    min_node, max_node = slider_value

    # Map node ids to labels
    node_labels = get_node_labels_from_nodes_df(nodes_df)

    # Check if we should display a subgraph
    is_subgraph = (ctx.triggered_id != "graph-selector" and data is not None and data['id'] == selected_graph_id) or ctx.triggered_id == 'jaccard-bar-chart'

    # Get the node and edges data from the subgraph (this is only for the subgraph) and it could be a jaccard graph
    if ctx.triggered_id == 'jaccard-bar-chart' and jaccard_click_data != None:
        word = jaccard_click_data["points"][0]["label"]
        source_data = jaccard_data.get(word, {})
    elif data and data['id'] == selected_graph_id:
        source_data = data
    else:
        source_data = {}

    sub_node_data = pd.DataFrame(source_data.get('nodes_df', []))
    sub_edges_data = pd.DataFrame(source_data.get('edges_df', []))


    # Lookup sets for styling (only meaningful for subgraph)
    sub_nodes_set = set(sub_node_data['node_id']) if is_subgraph else set()
    sub_edges_set = {
        (row['src'], row['edge_attr'], row['dst'])
        for row in sub_edges_data.to_dict('records')
    } if is_subgraph else set()
    # Build node elements for cytoscape
    node_elements = []
    node_items = list(node_labels.items())
    valid_nodes = set()
    for i in range(min_node, max_node):
        node_id, label = node_items[i]
        # Skip any nodes that are not in the sub_graph is we want to view sub_graph only
        if subgraph_tick_value and int(node_id) not in sub_nodes_set:
            continue
        # append the node to the graph
        node_elements.append({
                "data": {"id": str(node_id), "label": label},
                "classes": "subgraph-node" if int(node_id) in sub_nodes_set else "basic-node"
            })
        valid_nodes.add(node_id)
    # Build edge elements for cytoscape
    edge_elements = [
        {
            "data": {
                "source": str(row['src']),
                "target": str(row['dst']),
                "label": str(row['edge_attr'])
            },
            "classes": "subgraph-edge" if (row['src'], row['edge_attr'], row['dst']) in sub_edges_set else "basic-edge"
        }
        for _, row in edges_df.iterrows()
        if (str(row['src']) in valid_nodes and 
            str(row['dst']) in valid_nodes and 
            (row['src'] != row['dst'] or show_self_loops))
    ]

    elements = node_elements + edge_elements

    stylesheet = get_styles()

    wipe_data = ctx.triggered_id == "graph-selector"

    return elements, stylesheet, min_value, max_value, marks, slider_value, wipe_data

def update_node_slider(nodes):
    min_id = 0
    max_id = len(nodes)

    step = max(1, (max_id - min_id) // 10)
    marks = {i: str(i) for i in range(min_id, max_id + 1, step)}

    max_value = len(nodes) if len(nodes) < 50 else 50

    return min_id, max_id, marks, [min_id, max_value]

#######
## Callback for running the query and getting output/subgraph
#######
@app.callback(
    Output('trigger-loading', 'children'),
    Output('query-output-data', 'data'),
    Output("subgraph-selection", "data"),
    Input('run-query-button', 'n_clicks'),
    State('query-text', 'value'),
    State('graph-selector', 'value'),
    State('dataset-selector', 'value'),
    State('compute-jaccard', 'value'),
    State('seed-input', 'value'),
    prevent_initial_call=True, # To make sure we do not load this whole callback when the app loads
    running=[
        (Output("run-query-button", "disabled"), True, False),
        (Output("graph-selector", "disabled"), True, False),
        (Output("dataset-selector", "disabled"), True, False)
    ]
)
def run_query(n_clicks, query, selected_graph_id, selected_dataset_name, compute_jaccard, seed):
    if not query or not selected_graph_id or not selected_dataset_name:
        return "Not all input was provided", False

    dataset_loader = LOADED_DATASETS[selected_dataset_name]

    class Args:
        sample_idx = selected_graph_id
        dataset = selected_dataset_name
    
    args = Args()
    args.query = query
    args.seed = seed

    # get the embedding model etc from the dataset loader
    emb_model = dataset_loader.emb_model
    emb_tokenizer = dataset_loader.emb_tokenizer
    emb_device = dataset_loader.emb_device
    text2embedding = dataset_loader.text2embedding

    # Get the subgraph and description
    sub_graph, desc = get_subgraph(args, dataset_loader.dataset_module, emb_model, emb_tokenizer, emb_device, query, text2embedding)

    # Get the subgraph for cytoscape to annotate the subgraph gotten from the query
    nodes_df, edges_df = graph_desc_to_graph(desc)

    # Turn it into data to send it to another callback
    data = {
        'id': selected_graph_id,
        'nodes_df': nodes_df.to_dict('records'),
        'edges_df': edges_df.to_dict('records')
    }

    # Query the LLM
    _, llm_answer = predict_graph(args, desc, query)

    # Compute jaccard values
    jaccard_info = {} # if we do not compute it, we return an empty dictionairy
    if compute_jaccard:
        graph_display = {'display': 'block'}
        query_combinations = multiple_queries_clauses(args.query)
        jaccard_info = {}
        jaccard_data = {} # the data that will contain the subgraphs, that can then be displayed on the webpage
        for word, temp_query in query_combinations.items():
            args.query = temp_query # Change the query to the new query to get a different subgraph
            temp_graph, temp_desc = get_subgraph(args, 
                                dataset_loader.dataset_module, 
                                emb_model, 
                                emb_tokenizer, 
                                emb_device, 
                                temp_query, 
                                text2embedding)
            # Dictionairy of key being the missing word, and the value a tuple of the similarity value and subgraph
            similarity = jaccard_similarity(sub_graph, temp_graph)
            jaccard_info[word] = (1 - similarity, temp_desc) # Invert the score by doing 1 - similarity

            # Get the subgraph data in a way for cytoscape to display it
            temp_nodes_df, temp_edges_df = graph_desc_to_graph(temp_desc)

            # Turn it into data to be used by other calls
            temp_data = {
                'nodes_df': temp_nodes_df.to_dict('records'),
                'edges_df': temp_edges_df.to_dict('records')
            }
            jaccard_data[word] = temp_data
    else:
        graph_display = {'display': 'none'}
        jaccard_data = None # the data that will contain the subgraphs, that can then be displayed on the webpage
    
    # Create the jaccard figure (we dont display it due to display : none if jaccard was off)
    filtered_jaccard_info = {k: v for k, v in jaccard_info.items() if v[0] != 0} # Filter out entries with a Jaccard score of 0 (the bar chart otherwise is confusing)

    # Extract scores and corresponding labels
    scores = [val[0] for val in filtered_jaccard_info.values()]
    labels = list(filtered_jaccard_info.keys())
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=labels,
            orientation='h',
            text=[f"{s:.2f}" for s in scores],
            textposition='auto',
            marker_color='teal',
            hoverinfo='x+y'
        )
    ])

    fig.update_layout(
        margin=dict(l=80, r=10, t=10, b=40),
        yaxis=dict(title=''),
        xaxis=dict(title='Significance score'),
        height=300,
        plot_bgcolor='white',
        clickmode='event+select'
    )

    return " ", {"answer": llm_answer, "jaccard": graph_display, "figure": fig, "graph_data": jaccard_data}, data


#######
## Callback for setting the query output, having it in a seperate callback allows us to wipe it aswell
#######
@app.callback(
    Output('query-output', 'children'),
    Output('jaccard-bar-container', 'style'),
    Output('jaccard-bar-chart', 'figure'),
    Output('jaccard-graph', 'data'),
    Output('node-info-text-area', 'value'),
    Input('query-output-data', 'data'),
    Input('query-wipe-data-select', 'data'),
    Input('query-wipe-graph-select', 'data'),
    Input('node-info-data', 'data'),
    State('query-output', 'children'),
    State('jaccard-bar-container', 'style'),
    State('jaccard-bar-chart', 'figure'),
    State('jaccard-graph', 'data'),
    State('node-info-data', 'data'),
    prevent_initial_call=True,
)
def set_query_output(query_data, _, graph_wipe, node_info_data, current_text, current_style, current_figure, current_graph_data, current_node_data):
    if ctx.triggered_id == "query-output-data": # if query text was provided
        return query_data['answer'], query_data['jaccard'], query_data['figure'], query_data['graph_data'], current_node_data
    if ctx.triggered_id == "node-info-data": # if node info was provided
        return current_text, current_style, current_figure, current_graph_data, node_info_data
    elif graph_wipe or ctx.triggered_id == "query-wipe-data-select": # if a wipe is needed
        return "", {'display': 'none'}, go.Figure(), {}, ""
    else:
        return current_text, current_style, current_figure, current_graph_data, "" # just keep what we had if nothing matched
#######
## Callback for showing the node info when clicking on a node
#######
@app.callback(
    Output('node-info-data', 'data'),
    Input('cytoscape-graph', 'tapNode'),
    State('cytoscape-graph', 'elements'),
    prevent_initial_call=True,
)
def displayTapNodeData(data, graph_data):
    tapped_node_id = data['data']['id']
    tapped_edges_data = data.get('edgesData', [])
    node_labels = {item['data']['id']: item['data']['label'] for item in graph_data if 'source' not in item['data'] and 'target' not in item['data']}

    outgoing_edges = []
    incoming_edges = []
    tapped_node_label = node_labels[tapped_node_id]
    for edge in tapped_edges_data:
        if edge['source'] == tapped_node_id:
            target_id = edge['target']
            edge_label = edge.get('label', '')
            target_label = node_labels.get(target_id, '[Unknown]')
            outgoing_edges.append(f"{tapped_node_label} → {edge_label} → {target_label}")
        else:
            target_id = edge['source']
            edge_label = edge.get('label', '')
            target_label = node_labels.get(target_id, '[Unknown]')
            incoming_edges.append(f"{target_label} → {edge_label} → {tapped_node_label}")

    output = (
        f"Node: {tapped_node_label}\n"
        "-------------------------\n"
        "Outgoing edges:\n"
        + ("\n".join(outgoing_edges) if outgoing_edges else "  (None)") + "\n\n"
        "Incoming edges:\n"
        + ("\n".join(incoming_edges) if incoming_edges else "  (None)")
    )

    return output

if __name__ == '__main__':
    # Nltk downloading for the jaccard values (only downloads if you do not have them already)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    #seed_everything(seed=args.seed)
    app.run(debug=True)
