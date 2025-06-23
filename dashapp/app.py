import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import plotly.express as px

app = dash.Dash(__name__)
#######
## Color related helper stuff
#######
color_scale = [
    "#0000FF",
    "#3300CC",
    "#660099",
    "#990066",
    "#CC0033",
    "#FF0000"
]

def attention_to_color(value):
    index = int(value * (len(color_scale) - 1))
    return color_scale[index]

#######
## Mock data definitions
#######

DATASETS = {
    "Dataset 1": {
        "Graph A": {
            "nodes": [
                {"data": {"id": "a", "label": "A", "attention": 0.9}},
                {"data": {"id": "b", "label": "B", "attention": 0.3}},
                {"data": {"id": "c", "label": "C", "attention": 0.6}},
            ],
            "edges": [
                {"data": {"source": "a", "target": "b"}},
                {"data": {"source": "a", "target": "c"}},
            ]
        },
        "Graph B": {
            "nodes": [
                {"data": {"id": "d", "label": "D", "attention": 0.2}},
                {"data": {"id": "e", "label": "E", "attention": 0.5}},
            ],
            "edges": [
                {"data": {"source": "d", "target": "e"}},
            ]
        }
    },
    "Dataset 2": {
        "Graph X": {
            "nodes": [
                {"data": {"id": "x", "label": "X", "attention": 0.1}},
                {"data": {"id": "y", "label": "Y", "attention": 0.8}},
            ],
            "edges": [
                {"data": {"source": "x", "target": "y"}},
            ]
        },
        "Graph Y": {
            "nodes": [
                {"data": {"id": "z", "label": "Z", "attention": 0.5}},
                {"data": {"id": "w", "label": "W", "attention": 0.7}},
            ],
            "edges": [
                {"data": {"source": "z", "target": "w"}},
            ]
        }
    }
}

#######
## Helper function to get the style of the nodes based on their attention (in here i do for example size and colloring)
## At the moment i still got some "hardcoded" stuff in here, such as min and max size being hardset
#######
def get_node_styles(nodes, show_attention=True, threshold=0.0):
    min_size = 50
    max_size = 100
    styles = []

    for node in nodes:
        attention = node['data']['attention']
        node_id = node['data']['id']
        label = node['data']['label']
        off = attention < threshold

        styles.append({
            'selector': f'node[id = "{node_id}"]',
            'style': {
                'label': '' if off else f"{label} ({attention:.2f})",
                'background-color': attention_to_color(attention) if (show_attention and not off) else "#ccc",
                'width': min_size + (max_size - min_size) * attention if not off else min_size,
                'height': min_size + (max_size - min_size) * attention if not off else min_size,
                'font-size': 12,
                'color': "#fff",
                'text-valign': 'center',
                'text-halign': 'center',
                'opacity': 0.2 if off else 1.0
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
    html.Div([
        html.H3("Dataset"),
        dcc.Dropdown(
            id='dataset-selector',
            options=[{"label": name, "value": name} for name in DATASETS.keys()],
            value='Dataset 1',
            clearable=False
        ),
        html.H3("Graph"),
        dcc.Dropdown(id='graph-selector', clearable=False),
        html.H3("Query"),
        dcc.Textarea(
            id='query-text',
            value='write here to query G-retriever.',
            style={'width': '100%', 'height': 150}
        ),
        html.Button('Run Query', id='run-query-button', n_clicks=0, style={'marginTop': '10px'}),
        dcc.Checklist(
            id='toggle-controls',
            options=[
                {'label': 'Toggle attention color', 'value': 'attention'},
                {'label': 'Toggle legend', 'value': 'legend'}
            ],
            value=['attention', 'legend'],
            style={'marginTop': '10px'}
        )
    ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

    

    #######
    ## Graph output
    #######
    html.Div([
        html.H3("Subgraph"),
        cyto.Cytoscape(
            id='cytoscape-graph',
            layout={'name': 'breadthfirst'},
            style={'width': '100%', 'height': '400px'},
        ),

        html.H4("Query Output"),
        #######
        ## Query output box
        #######
        html.Div(
            id='query-output',
            style={
                'border': '1px solid #ccc',
                'padding': '10px',
                'minHeight': '60px',
                'marginTop': '10px',
                'whiteSpace': 'pre-wrap',
                'backgroundColor': '#f9f9f9'
            }
        )
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top'}),


    #######
    ## Legend and slider
    #######
    html.Div([
        html.H3("Legend"),

        html.Div([
            #######
            ## Marker on the legend
            #######
            html.Div(id='threshold-marker', children='●', style={
                'position': 'absolute',
                'left': '35px',
                'transform': 'translateX(-50%)',
                'fontSize': '14px',
                'color': 'black',
            }),

            #######
            ## Gradient effect for the legend and the bar
            #######
            html.Div(style={
                'height': '300px',
                'width': '30px',
                'background': 'linear-gradient(to top, ' + ', '.join(color_scale) + ')',
                'margin': '0 auto',
            }),
        ], style={
            'position': 'relative',
            'height': '300px',
            'margin': '20px auto',
            'width': '60px',
        }),

        #######
        ## Threshold slider
        #######
        html.Div(id='threshold-value', style={'textAlign': 'center', 'marginTop': '5px', 'fontSize': '14px'}),

        dcc.Slider(
            id='attention-threshold',
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.0,
            marks={0.0: '0.0', 0.5: '0.5', 1.0: '1.0'},
            tooltip={'placement': 'bottom'}
        )
    ], id='legend-container', style={'width': '20%', 'display': 'inline-block', 'padding': '10px'})

])

#######
## Callback for updating the list of all graphs based on the dataset that is currently active
#######
@app.callback(
    Output('graph-selector', 'options'),
    Output('graph-selector', 'value'),
    Input('dataset-selector', 'value')
)
def update_graph_list(dataset_name):
    graphs = DATASETS[dataset_name]
    options = [{"label": g, "value": g} for g in graphs.keys()]
    return options, options[0]['value']

#######
## Callback for showing the current graph (center element of the page), based on the currently selected graph from the dropdown
#######
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Output('cytoscape-graph', 'stylesheet'),
    Input('graph-selector', 'value'),
    Input('dataset-selector', 'value'),
    Input('toggle-controls', 'value'),
    Input('attention-threshold', 'value')
)

def update_graph(graph_name, dataset_name, toggles, threshold):
    if not graph_name:
        return [], []
    
    # Show attention colow based on the tickbox for it
    show_attention = 'attention' in toggles
    graph = DATASETS[dataset_name][graph_name]
    nodes = graph["nodes"]
    edges = graph["edges"]

    # Get all nodes that should be shown based on the threshold
    dimmed_node_ids = {node['data']['id'] for node in nodes if node['data']['attention'] < threshold}

    # Filter out any edges that should not be shown due to their nodes not being active due to to low threshold
    filtered_edges = [
        edge for edge in edges
        if edge['data']['source'] not in dimmed_node_ids and edge['data']['target'] not in dimmed_node_ids
    ]

    return nodes + filtered_edges, get_node_styles(nodes, show_attention, threshold)


#######
## Showing and hiding the legend
#######
@app.callback(
    Output('legend-container', 'style'),
    Input('toggle-controls', 'value'),
    State('legend-container', 'style')
)
def toggle_legend(toggles, current_style):
    current_style = current_style.copy()
    current_style['display'] = 'inline-block' if 'legend' in toggles else 'none'
    return current_style

#######
## Adjusting the legend threshold value
#######
@app.callback(
    Output('threshold-value', 'children'),
    Input('attention-threshold', 'value')
)
def update_threshold_label(value):
    return f"Current threshold: {value:.2f}"

#######
## The marker position on the legend
#######
@app.callback(
    Output('threshold-marker', 'style'),
    Input('attention-threshold', 'value'),
)
def update_marker_position(threshold):
    top_px = 300 * (1 - threshold)
    return {
        'position': 'absolute',
        'left': '35px',
        'top': f'{top_px}px',
        'transform': 'translate(-50%, -50%)',
        'fontSize': '14px',
        'color': 'black',
    }


if __name__ == '__main__':
    app.run(debug=True)
