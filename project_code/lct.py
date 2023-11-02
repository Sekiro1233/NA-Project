from graphviz import Digraph

dot = Digraph(
    comment="The Round Table",
    format="png",
    graph_attr={"rankdir": "LR", "ranksep": "0.7"},
)

# 创建节点
dot.node("1", "Import necessary libraries", shape="box", height="0.5", width="0.5")
dot.node(
    "1.1", "Define sparse_sym_matrix function", shape="box", height="0.5", width="0.5"
)
dot.node("1.1.1", "Generate random matrix", shape="box", height="0.5", width="0.5")
dot.node("1.1.2", "Sparse the matrix", shape="box", height="0.5", width="0.5")
dot.node("1.1.3", "Symmetrize the matrix", shape="box", height="0.5", width="0.5")
dot.node("1.2", "Define power_method function", shape="box", height="0.5", width="0.5")
dot.node("1.3", "Define qr_iteration function", shape="box", height="0.5", width="0.5")
dot.node(
    "1.4", "Define arnoldi_iteration1 function", shape="box", height="0.5", width="0.5"
)
dot.node(
    "1.5", "Define arnoldi_iteration2 function", shape="box", height="0.5", width="0.5"
)
dot.node("2", "Define MainWindow class", shape="box", height="0.5", width="0.5")
dot.node("2.1", "Define run_code method", shape="box", height="0.5", width="0.5")
dot.node("2.1.1", "Get user input", shape="box", height="0.5", width="0.5")
dot.node(
    "2.1.2",
    "Call corresponding function to execute calculation",
    shape="box",
    height="0.5",
    width="0.5",
)
dot.node("2.1.3", "Display results", shape="box", height="0.5", width="0.5")
dot.node("2.2", "Define open_readme method", shape="box", height="0.5", width="0.5")
dot.node(
    "2.2.1", "Open documentation on GitHub", shape="box", height="0.5", width="0.5"
)
dot.node(
    "3",
    "Create QApplication object and MainWindow object",
    shape="box",
    height="0.5",
    width="0.5",
)
dot.node("3.1", "Display GUI window", shape="box", height="0.5", width="0.5")

# 创建边，连接节点
dot.edges(
    [
        ("1", "1.1"),
        ("1.1", "1.1.1"),
        ("1.1", "1.1.2"),
        ("1.1", "1.1.3"),
        ("1", "1.2"),
        ("1", "1.3"),
        ("1", "1.4"),
        ("1", "1.5"),
        ("2", "2.1"),
        ("2.1", "2.1.1"),
        ("2.1", "2.1.2"),
        ("2.1", "2.1.3"),
        ("2", "2.2"),
        ("2.2", "2.2.1"),
        ("3", "3.1"),
    ]
)

# 保存和显示图
dot.view()
