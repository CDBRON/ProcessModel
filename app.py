import streamlit as st
import os
import json
import re
import sys
import datetime
import io
import contextlib
from typing import Union, List, Dict, cast, Any, TypedDict
from dataclasses import dataclass, field


# --- 新增绘图相关导入 ---
import networkx as nx
from lxml import etree
import pydot
import xml.etree.ElementTree as ET
from graphviz import Digraph
import textwrap
import html

# --- LangChain & AI Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langgraph.graph import StateGraph, END
import tiktoken

# ==========================================
# 1. Streamlit 页面配置与工具函数
# ==========================================
st.set_page_config(page_title="BPMN Process Generator Agent", layout="wide")


class StreamlitCapture(io.StringIO):
    """用于捕获 stdout 并实时更新到 Streamlit 界面"""

    def __init__(self, container):
        super().__init__()
        self.container = container
        self.text = ""

    def write(self, s):
        self.text += s
        # 实时更新容器内容
        self.container.code(self.text, language="text")

    def flush(self):
        pass




# ==========================================
# 2. BPMN 转换与绘图核心类 (新增部分)
# ==========================================

# --- 配置常量 ---
NS_MAP = {
    None: "http://www.omg.org/spec/BPMN/20100524/MODEL",
    'bpmndi': "http://www.omg.org/spec/BPMN/20100524/DI",
    'omgdi': "http://www.omg.org/spec/DD/20100524/DI",
    'omgdc': "http://www.omg.org/spec/DD/20100524/DC",
    'xsi': "http://www.w3.org/2001/XMLSchema-instance"
}

SIZES = {
    'task': {'w': 100, 'h': 80},
    'gateway': {'w': 50, 'h': 50},
    'event': {'w': 36, 'h': 36}
}

class BpmnConverter:
    def __init__(self, ir_json: Dict[str, Any]):
        self.ir = self._sanitize_ir(ir_json)
        self.graph = nx.DiGraph()
        self.elements = {}
        self.sequence_flows = []
        self.layout = {}
        self._id_counter = 0

    def _sanitize_ir(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        seen_ids = set()
        def visit(element):
            original_id = element.get('id')
            if original_id in seen_ids:
                new_id = f"{original_id}_{len(seen_ids)}"
                element['id'] = new_id
            seen_ids.add(element['id'])
            if 'branches' in element:
                for branch in element['branches']:
                    for seq_elem in branch.get('sequence', []):
                        visit(seq_elem)
        for root_element in ir_data.get('process', []):
            visit(root_element)
        return ir_data

    def _generate_unique_id(self, prefix: str) -> str:
        self._id_counter += 1
        return f"{prefix}_{self._id_counter}"

    def _get_element_size(self, elem_type: str):
        if 'gateway' in elem_type.lower():
            return SIZES['gateway']
        elif 'event' in elem_type.lower():
            return SIZES['event']
        else:
            return SIZES['task']

    def _parse_and_build_graph(self, sequence: List[Dict[str, Any]], incoming_node_id: str) -> str:
        current_node_id = incoming_node_id
        for element in sequence:
            elem_id = element['id']
            elem_type = element['type']

            if elem_id not in self.elements:
                self.elements[elem_id] = {
                    'type': elem_type,
                    'name': element.get('description', ''),
                }
                self.graph.add_node(elem_id)

            if current_node_id:
                if not self.graph.has_edge(current_node_id, elem_id):
                    self.graph.add_edge(current_node_id, elem_id)
                    flow_id = self._generate_unique_id("Flow")
                    self.sequence_flows.append({
                        'id': flow_id,
                        'source': current_node_id,
                        'target': elem_id,
                        'name': element.get('condition', '')
                    })

            if 'branches' in element:
                branch_endpoints = []
                for branch in element['branches']:
                    branch_end_id = self._parse_and_build_graph(branch.get('sequence', []), elem_id)
                    branch_endpoints.append(branch_end_id)
                    if branch.get('sequence'):
                        first_node_in_branch = branch['sequence'][0]['id']
                        for flow in self.sequence_flows:
                            if flow['source'] == elem_id and flow['target'] == first_node_in_branch:
                                flow['name'] = branch.get('condition', '')
                                break

                if len(branch_endpoints) > 1:
                    merge_id = self._generate_unique_id(f"Gateway_Merge_{elem_id}")
                    origin_type = element.get('type', '').lower()
                    merge_type = 'exclusiveGateway'
                    if 'parallel' in origin_type: merge_type = 'parallelGateway'
                    elif 'inclusive' in origin_type: merge_type = 'inclusiveGateway'

                    self.elements[merge_id] = {'type': merge_type, 'name': ''}
                    self.graph.add_node(merge_id)

                    for end_id in branch_endpoints:
                        if end_id and not self.graph.has_edge(end_id, merge_id):
                            self.graph.add_edge(end_id, merge_id)
                            flow_id = self._generate_unique_id("Flow")
                            self.sequence_flows.append({'id': flow_id, 'source': end_id, 'target': merge_id})
                    current_node_id = merge_id
                elif branch_endpoints:
                    current_node_id = branch_endpoints[0]
                else:
                    current_node_id = elem_id
            else:
                current_node_id = elem_id
        return current_node_id

    def _calculate_layout_with_graphviz(self):
        print("--- [Layout] Calculating positions... ---")
        dot_graph = pydot.Dot(graph_type='digraph', rankdir='LR', splines='ortho', nodesep='0.8', ranksep='1.2')

        for node_id, details in self.elements.items():
            size = self._get_element_size(details['type'])
            w_in = size['w'] / 72.0
            h_in = size['h'] / 72.0
            node = pydot.Node(node_id, width=w_in, height=h_in, shape='box', fixedsize=True)
            dot_graph.add_node(node)

        for flow in self.sequence_flows:
            dot_graph.add_edge(pydot.Edge(flow['source'], flow['target']))

        try:
            layout_json = json.loads(dot_graph.create(prog='dot', format='json'))
        except Exception as e:
            print(f"Graphviz Error: {e}. Ensure Graphviz is installed and in PATH.")
            # Fallback or exit handled by caller
            return

        bb = layout_json.get('bb', '0,0,1000,1000').split(',')
        canvas_height = float(bb[3])

        for obj in layout_json.get('objects', []):
            node_id = obj.get('name')
            if node_id in self.elements:
                pos = obj.get('pos', '0,0').split(',')
                x, y = float(pos[0]), float(pos[1])
                size = self._get_element_size(self.elements[node_id]['type'])
                bpmn_x = x - (size['w'] / 2)
                bpmn_y = canvas_height - y - (size['h'] / 2)

                self.layout[node_id] = {
                    'x': int(bpmn_x), 'y': int(bpmn_y),
                    'width': size['w'], 'height': size['h']
                }
        self._calculate_orthogonal_edges()

    def _calculate_orthogonal_edges(self):
        for flow in self.sequence_flows:
            source_id = flow['source']
            target_id = flow['target']
            if source_id not in self.layout or target_id not in self.layout: continue

            src = self.layout[source_id]
            tgt = self.layout[target_id]
            start_x = src['x'] + src['width']
            start_y = src['y'] + (src['height'] // 2)
            end_x = tgt['x']
            end_y = tgt['y'] + (tgt['height'] // 2)

            waypoints = [{'x': start_x, 'y': start_y}]
            mid_x = int((start_x + end_x) / 2)

            if abs(start_y - end_y) < 2: pass
            elif start_x < end_x:
                waypoints.append({'x': mid_x, 'y': start_y})
                waypoints.append({'x': mid_x, 'y': end_y})
            else:
                safe_y = max(src['y'] + src['height'], tgt['y'] + tgt['height']) + 20
                waypoints.append({'x': start_x + 20, 'y': start_y})
                waypoints.append({'x': start_x + 20, 'y': safe_y})
                waypoints.append({'x': end_x - 20, 'y': safe_y})
                waypoints.append({'x': end_x - 20, 'y': end_y})
            waypoints.append({'x': end_x, 'y': end_y})
            flow['waypoints'] = waypoints

            if flow.get('name'):
                p1 = waypoints[0]
                p2 = waypoints[1] if len(waypoints) > 1 else waypoints[0]
                label_x = (p1['x'] + p2['x']) / 2
                label_y = (p1['y'] + p2['y']) / 2
                label_w, label_h = 80, 20
                if abs(p1['y'] - p2['y']) < 5:
                    label_y -= 15
                    label_x -= (label_w / 2)
                else:
                    label_x += 5
                    label_y -= (label_h / 2)
                flow['label_bounds'] = {'x': int(label_x), 'y': int(label_y), 'width': label_w, 'height': label_h}

    def _build_xml(self) -> str:
        definitions = etree.Element('definitions', nsmap=NS_MAP, id="Definitions_1", targetNamespace=NS_MAP[None])
        process = etree.SubElement(definitions, 'process', id="Process_1", isExecutable="false")

        for elem_id, details in self.elements.items():
            elem_type = details['type']
            name = details.get('name', '')
            tag = 'task'
            if 'start' in elem_type.lower(): tag = 'startEvent'
            elif 'end' in elem_type.lower(): tag = 'endEvent'
            elif 'gateway' in elem_type.lower():
                tag = 'parallelGateway' if 'parallel' in elem_type.lower() else 'exclusiveGateway'
            node = etree.SubElement(process, tag, id=elem_id)
            if name: node.set('name', name)
            for flow in self.sequence_flows:
                if flow['source'] == elem_id: etree.SubElement(node, 'outgoing').text = flow['id']
                if flow['target'] == elem_id: etree.SubElement(node, 'incoming').text = flow['id']

        for flow in self.sequence_flows:
            seq_flow = etree.SubElement(process, 'sequenceFlow', id=flow['id'], sourceRef=flow['source'], targetRef=flow['target'])
            if flow.get('name'): seq_flow.set('name', flow['name'])

        diagram = etree.SubElement(definitions, etree.QName(NS_MAP['bpmndi'], 'BPMNDiagram'), id="BPMNDiagram_1")
        plane = etree.SubElement(diagram, etree.QName(NS_MAP['bpmndi'], 'BPMNPlane'), id="BPMNPlane_1", bpmnElement="Process_1")

        for elem_id, bounds in self.layout.items():
            shape = etree.SubElement(plane, etree.QName(NS_MAP['bpmndi'], 'BPMNShape'), bpmnElement=elem_id, id=f"{elem_id}_di")
            etree.SubElement(shape, etree.QName(NS_MAP['omgdc'], 'Bounds'), x=str(bounds['x']), y=str(bounds['y']), width=str(bounds['width']), height=str(bounds['height']))
            if self.elements[elem_id].get('name'):
                label = etree.SubElement(shape, etree.QName(NS_MAP['bpmndi'], 'BPMNLabel'))
                etree.SubElement(label, etree.QName(NS_MAP['omgdc'], 'Bounds'), x=str(bounds['x']), y=str(bounds['y'] + bounds['height']), width=str(bounds['width']), height="14")

        for flow in self.sequence_flows:
            if 'waypoints' not in flow: continue
            edge = etree.SubElement(plane, etree.QName(NS_MAP['bpmndi'], 'BPMNEdge'), bpmnElement=flow['id'], id=f"{flow['id']}_di")
            for pt in flow['waypoints']:
                etree.SubElement(edge, etree.QName(NS_MAP['omgdi'], 'waypoint'), x=str(pt['x']), y=str(pt['y']))
            if flow.get('name') and 'label_bounds' in flow:
                lb = flow['label_bounds']
                label_elem = etree.SubElement(edge, etree.QName(NS_MAP['bpmndi'], 'BPMNLabel'))
                etree.SubElement(label_elem, etree.QName(NS_MAP['omgdc'], 'Bounds'), x=str(lb['x']), y=str(lb['y']), width=str(lb['width']), height=str(lb['height']))

        return etree.tostring(definitions, pretty_print=True, xml_declaration=True, encoding='UTF-8').decode('utf-8')

    def convert(self) -> str:
        self._parse_and_build_graph(self.ir['process'], None)
        start_nodes = [n for n, d in self.graph.in_degree() if d == 0]
        end_nodes = [n for n, d in self.graph.out_degree() if d == 0]

        if not any(self.elements[n]['type'] == 'startEvent' for n in start_nodes):
            sid = "StartEvent_1"
            self.elements[sid] = {'type': 'startEvent', 'name': 'Start'}
            for node in start_nodes:
                fid = self._generate_unique_id("Flow")
                self.sequence_flows.insert(0, {'id': fid, 'source': sid, 'target': node})

        if not any(self.elements[n]['type'] == 'endEvent' for n in end_nodes):
            eid = "EndEvent_1"
            self.elements[eid] = {'type': 'endEvent', 'name': 'End'}
            for node in end_nodes:
                fid = self._generate_unique_id("Flow")
                self.sequence_flows.append({'id': fid, 'source': node, 'target': eid})

        self._calculate_layout_with_graphviz()
        return self._build_xml()

def remove_role_prefix_from_bpmn(xml_string: str) -> str:
    """移除 XML 中的角色前缀"""
    xml_string = xml_string.strip()
    try:
        namespaces = {}
        ns_matches = re.finditer(r'\s+xmlns:?(?P<prefix>\w+)?="(?P<uri>[^"]+)"', xml_string)
        for match in ns_matches:
            prefix = match.group('prefix') or ''
            uri = match.group('uri')
            namespaces[prefix] = uri
            ET.register_namespace(prefix, uri)

        ns_map_for_find = {prefix: uri for prefix, uri in namespaces.items() if prefix}
        if '' in namespaces: ns_map_for_find['bpmn'] = namespaces['']

        root = ET.fromstring(xml_string)
        elements_to_process = []
        tags_to_check = ['task', 'exclusiveGateway', 'parallelGateway', 'inclusiveGateway']

        for tag in tags_to_check:
            path = f".//bpmn:{tag}" if 'bpmn' in ns_map_for_find else f".//{tag}"
            elements_to_process.extend(root.findall(path, ns_map_for_find))

        for element in elements_to_process:
            name_attr = element.get('name')
            if name_attr and ':' in name_attr:
                parts = name_attr.split(':', 1)
                if len(parts) > 1 and parts[0].strip() != '':
                    element.set('name', parts[1].strip())

        return "<?xml version='1.0' encoding='UTF-8'?>\n" + ET.tostring(root, encoding='unicode')
    except Exception as e:
        print(f"XML Cleaning Error: {e}")
        return xml_string

def bpmn_to_svg(xml_content, output_filename='bpmn_process_hd'):
    """将 BPMN XML 转换为 SVG 并返回文件路径"""
    xml_content = xml_content.replace('xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"', '')
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return None

    dot = Digraph(comment='BPMN Process', format='svg')
    dot.attr(dpi='300')
    dot.attr(rankdir='LR')
    dot.attr('node', fontname='Arial')
    dot.attr(forcelabels='true')
    dot.attr(nodesep='0.8')
    dot.attr(ranksep='1.0')

    def create_xlabel(text, fontsize):
        if not text: return ""
        wrapped_lines = textwrap.wrap(text, width=12)
        escaped_lines = [html.escape(line) for line in wrapped_lines]
        joined_text = "<BR/>".join(escaped_lines)
        return f'<<FONT POINT-SIZE="{fontsize}">{joined_text}</FONT>>'

    def add_node(node_id, label, node_type):
        if 'startEvent' in node_type:
            dot.node(node_id, 'Start', shape='circle', style='filled', fillcolor='#E6FFCC', width='0.6', fixedsize='true', fontsize='10')
        elif 'endEvent' in node_type:
            dot.node(node_id, 'End', shape='doublecircle', style='filled', fillcolor='#FFCCCC', width='0.6', fixedsize='true', penwidth='3', fontsize='10')
        elif 'Gateway' in node_type:
            symbol = '?'
            if 'parallel' in node_type: symbol = '+'
            elif 'inclusive' in node_type: symbol = 'O'
            elif 'exclusive' in node_type: symbol = 'X'
            xlabel_html = create_xlabel(label, 14)
            dot.node(node_id, symbol, shape='diamond', style='filled', fillcolor='#FFFFCC', fontsize='32', width='0.8', height='0.8', fixedsize='true', xlabel=xlabel_html)
        elif 'task' in node_type:
            wrapped_label = '\n'.join(textwrap.wrap(label, width=15)) if label else ""
            dot.node(node_id, wrapped_label, shape='box', style='rounded,filled', fillcolor='#E6F7FF', width='1.5', fontsize='12')
        else:
            dot.node(node_id, label if label else node_type, shape='box')

    process = root.find('process')
    if process is None:
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        process = root.find('bpmn:process', ns)
    if process is None: return None

    target_tags = ['task', 'userTask', 'serviceTask', 'scriptTask', 'businessRuleTask', 'manualTask', 'sendTask', 'receiveTask', 'exclusiveGateway', 'parallelGateway', 'inclusiveGateway', 'complexGateway', 'eventBasedGateway', 'startEvent', 'endEvent', 'intermediateCatchEvent', 'intermediateThrowEvent']

    for child in process:
        tag = child.tag.split('}')[-1]
        if tag in target_tags:
            add_node(child.get('id'), child.get('name', ''), tag)

    for child in process:
        tag = child.tag.split('}')[-1]
        if tag == 'sequenceFlow':
            dot.edge(child.get('sourceRef'), child.get('targetRef'), label=child.get('name', ''), fontsize='10')

    try:
        # 返回生成的 SVG 文件路径
        output_path = dot.render(output_filename, view=False, cleanup=True)
        return output_path
    except Exception as e:
        print(f"Graphviz Render Error: {e}")
        return None


# ==========================================
# 2. 核心类定义 (保持原逻辑)
# ==========================================

class TokenUsageCallbackHandler(BaseCallbackHandler):
    """本地估算版 Token 处理器"""

    def __init__(self, model_encoding="cl100k_base"):
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        try:
            self.encoder = tiktoken.get_encoding(model_encoding)
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        self._current_input_tokens = 0

    def _count_tokens(self, text: str) -> int:
        if not text: return 0
        try:
            return len(self.encoder.encode(text))
        except Exception:
            return int(len(text) * 0.7)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        count = 0
        for p in prompts:
            count += self._count_tokens(p)
        self._current_input_tokens = count

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = None
        if response.generations:
            generation = response.generations[0][0]
            if hasattr(generation, 'message') and hasattr(generation.message, 'response_metadata'):
                usage = generation.message.response_metadata.get(
                    'token_usage') or generation.message.response_metadata.get('usage')
            elif hasattr(generation, 'generation_info'):
                usage = generation.generation_info.get('token_usage') or generation.generation_info.get('usage')

        if not usage and response.llm_output:
            usage = response.llm_output.get('token_usage') or response.llm_output.get('usage')

        if usage:
            p_tokens = usage.get('prompt_tokens') or usage.get('input_tokens') or 0
            c_tokens = usage.get('completion_tokens') or usage.get('output_tokens') or 0
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
        else:
            p_tokens = self._current_input_tokens
            c_tokens = 0
            if response.generations:
                for gen_list in response.generations:
                    for gen in gen_list:
                        c_tokens += self._count_tokens(gen.text)
            self.prompt_tokens += p_tokens
            self.completion_tokens += c_tokens
            print(f"   [Source: Local Calc] In: {p_tokens}, Out: {c_tokens}")

    def get_and_reset_totals(self) -> Dict[str, int]:
        totals = {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._current_input_tokens = 0
        return totals


class SharedMemory:
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.costs: List[Dict[str, Any]] = []
        print("--- [Memory] Shared Memory module initialized. ---")

    def add_entry(self, agent_name: str, input_data: Any, output_data: Any, cost_info: Dict = None):
        timestamp = datetime.datetime.now().isoformat()
        self.log.append({
            "timestamp": timestamp,
            "agent_name": agent_name,
            "input": input_data,
            "output": output_data
        })
        print(f"--- [Memory] Logged entry for {agent_name}. ---")
        if cost_info:
            self.costs.append({"timestamp": timestamp, "agent_name": agent_name, **cost_info})

    def get_formatted_log(self) -> str:
        if not self.log: return "No interactions have been logged yet."
        formatted_entries = []
        for i, entry in enumerate(self.log):
            try:
                output_str = json.dumps(entry.get('output', ''), indent=2, ensure_ascii=False)
            except TypeError:
                output_str = str(entry.get('output', ''))
            if len(output_str) > 600: output_str = output_str[:600] + "\n... (output truncated)"

            try:
                input_str = json.dumps(entry.get('input', ''), indent=2, ensure_ascii=False)
            except TypeError:
                input_str = str(entry.get('input', ''))
            if len(input_str) > 400: input_str = input_str[:400] + "\n... (input truncated)"

            formatted_entry = (
                f"--- Log Entry {i + 1} ---\nAgent: {entry['agent_name']}\nTimestamp: {entry['timestamp']}\nInput:\n{input_str}\nOutput:\n{output_str}")
            formatted_entries.append(formatted_entry)
        return "\n\n".join(formatted_entries)


# ==========================================
# 3. 模型加载与工具定义
# ==========================================

@st.cache_resource
def load_activity_classifier(model_path):
    """加载本地分类模型，使用缓存避免重复加载"""
    # 注意：缓存函数内部不要使用 print 输出到 UI，否则会报错 CacheReplayClosureError
    if os.path.exists(model_path):
        try:
            # 这里去掉了 print 语句
            pipeline_instance = pipeline("text-classification", model=model_path, tokenizer=model_path)
            return pipeline_instance
        except Exception as e:
            # 如果出错，可以在这里抛出异常或者返回 None，但不要 print 到 UI
            return None
    else:
        return None


# 定义全局工具函数 (需要访问全局变量，但在 Streamlit 中我们将通过闭包或参数传递)
# 为了适配 LangChain 的 @tool 装饰器，我们需要在执行流程内部动态绑定或使用全局变量
# 这里我们使用全局变量占位符，在运行时更新

GLOBAL_TAVILY_KEY = ""
GLOBAL_CLIENT = None  # OpenAI Client


@tool
def tavily_wrapper(query: str) -> str:
    """
    A search engine that also provides a comprehensive, AI-generated answer based on the top search results.
    """
    print(f"\n--- [Tool Wrapper] Calling TavilySearch with query: {query} ---\n")
    if not GLOBAL_TAVILY_KEY:
        return "Error: Tavily API Key is missing."

    client = TavilyClient(api_key=GLOBAL_TAVILY_KEY)
    try:
        response = client.search(query=query, search_depth="advanced", include_answer="advanced", max_results=3)
    except Exception as e:
        return f"Tavily search failed: {e}"

    ai_generated_answer = response.get('answer', 'No direct answer was generated.')
    results = response.get('results', [])
    source_snippets = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            snippet = result.get('content', 'No snippet available.')
            source_snippets.append(f"Source {i + 1} Snippet: {snippet}")

    final_output = (
                f"**AI-Generated Summary Answer:**\n{ai_generated_answer}\n\n--- \n**Supporting Source Snippets:**\n" + "\n---\n".join(
            source_snippets))
    return final_output


@tool(return_direct=True)
def terminate(final_answer: str) -> str:
    """Use this tool as the very final action to stop execution."""
    print(f"\n--- [TERMINATE TOOL CALLED] ---")
    return final_answer


@tool(return_direct=True)
def finalize_process_design(enriched_json_str: str) -> str:
    """Finalize the enriched process design."""
    print("\n--- [GatewayInferenceAgent] Finalizing... ---\n")
    try:
        start_index = enriched_json_str.find('{')
        end_index = enriched_json_str.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            clean_json_str = enriched_json_str[start_index: end_index + 1]
            json_obj = json.loads(clean_json_str)
            return json.dumps(json_obj, indent=2, ensure_ascii=False)
        return "Error: Could not find valid JSON."
    except Exception as e:
        return f"Error: {e}"


@tool(return_direct=True)
def align_activity_granularity(input_data: Union[str, dict]) -> str:
    """Iteratively refines activities to 'Standard' granularity."""
    # 注意：这个工具依赖于 aligner_app，我们需要在运行时确保它可用
    # 这里为了简化，我们假设 aligner_app 已经在全局作用域构建完成
    # 在 Streamlit 中，这需要特别处理，见下文 run_workflow
    return "This tool logic is handled inside the graph execution context."


@tool(return_direct=True)
def deliver_abstracted_design(abstracted_json_str: str) -> str:
    """Deliver simplified JSON."""
    print("\n--- [PruningAbstractionAgent] Delivering... ---\n")
    try:
        start_index = abstracted_json_str.find('{')
        end_index = abstracted_json_str.rfind('}')
        if start_index != -1 and end_index != -1:
            clean_json_str = abstracted_json_str[start_index: end_index + 1]
            json_obj = json.loads(clean_json_str)
            return json.dumps(json_obj, indent=2, ensure_ascii=False)
        return "Error: No JSON found."
    except Exception as e:
        return f"Error: {e}"


@tool(return_direct=True)
def assemble_process_components(final_abstracted_json_str: str) -> str:
    """Assembles components into IR."""
    print("\n--- [AssemblyAgent] Starting assembly... ---\n")
    # 注意：ProcessAssembler 类依赖 GLOBAL_CLIENT
    try:
        match = re.search(r'{.*}', final_abstracted_json_str, re.DOTALL)
        if not match: raise ValueError("No JSON found")
        components_json = json.loads(match.group(0))

        # 实例化 Assembler (需要 LLM Client)
        assembler = ProcessAssembler(components_json, GLOBAL_CLIENT)
        final_ir = assembler.assemble()
        return json.dumps(final_ir, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


@tool(return_direct=True)
def finalize_audit(audited_ir_json_str: str) -> str:
    """Final audit step."""
    print("\n--- [FinalAuditorAgent] Delivering final IR... ---\n")
    try:
        start_index = audited_ir_json_str.find('{')
        end_index = audited_ir_json_str.rfind('}')
        if start_index == -1 or end_index == -1: raise ValueError("No JSON found")
        json_str = audited_ir_json_str[start_index: end_index + 1]
        # 简单清理
        cleaned_json_str = json_str.replace('"极description"', '"description"').replace('"type":极', '"type":')
        json_obj = json.loads(cleaned_json_str)
        return json.dumps(json_obj, indent=2, ensure_ascii=False)
    except Exception as e:
        return f"Error: {e}"


# --- Process Assembler Class ---
class ProcessAssembler:
    def __init__(self, components_json: dict, llm_client):
        self.llm = llm_client
        self.activities = {act['id']: act for act in components_json.get('activities', [])}
        self.gateways = {gate['id']: gate for gate in components_json.get('gateways', [])}
        self.final_ir = {"process": []}
        self.fix_counter = 0

    def _get_llm_decision(self, prompt: str) -> str:
        messages = [{"role": "system", "content": "You are a logical reasoning engine."},
                    {"role": "user", "content": prompt}]
        response = self.llm.chat.completions.create(
            model='Qwen/Qwen3-235B-A22B',
            messages=messages, temperature=0, stream=False
        )
        return response.choices[0].message.content.strip()

    def _find_start_activity_id(self) -> str:
        print("--- [Assembler] Finding start point...")
        descs = "\n".join([f"- {act['id']}: {act['description']}" for act in self.activities.values()])
        prompt = f"Identify start activity ID from:\n{descs}\nRespond ONLY with ID."
        start_id = self._get_llm_decision(prompt)
        match = re.search(r'act[\w_-]*', start_id)
        return match.group(0) if match else list(self.activities.keys())[0]

    def _find_next_happy_path_id(self, current_id: str, remaining_ids: list) -> Union[str, None]:
        if not remaining_ids: return None
        current_desc = self.activities[current_id]['description']
        rem_descs = "\n".join([f"- {aid}: {self.activities[aid]['description']}" for aid in remaining_ids])
        prompt = f"Last step: {current_desc}. Pick next logical step from:\n{rem_descs}\nRespond ID or None."
        next_id = self._get_llm_decision(prompt)
        if "None" in next_id: return None
        match = re.search(r'act[\w_-]*', next_id)
        return match.group(0) if match else None

    def build_process_trunk(self):
        start_id = self._find_start_activity_id()
        trunk_ids = []
        current_id = start_id
        remaining = set(self.activities.keys())
        while current_id and current_id in remaining:
            trunk_ids.append(current_id)
            remaining.remove(current_id)
            current_id = self._find_next_happy_path_id(current_id, list(remaining))
        self.final_ir['process'] = [{"type": "activity", "id": aid} for aid in trunk_ids]

    def graft_gateways(self):
        gateways = list(self.gateways.values())
        if not gateways: return
        prompt = f"""
        Integrate gateways into trunk.
        Trunk: {json.dumps(self.final_ir, indent=2)}
        Activities: {json.dumps(list(self.activities.values()), indent=2)}
        Gateways: {json.dumps(gateways, indent=2)}
        Output nested JSON process.
        """
        response = self.llm.chat.completions.create(
            model='XiaomiMiMo/MiMo-V2-Flash',
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        try:
            self.final_ir = json.loads(response.choices[0].message.content)
        except:
            print("Gateway grafting failed.")

    def assemble(self) -> dict:
        self.build_process_trunk()
        self.graft_gateways()
        return self.final_ir


# ==========================================
# 4. 工作流构建与执行逻辑
# ==========================================

@dataclass
class GraphState:
    input_json: Dict = field(default_factory=dict)
    processed_json: Dict = field(default_factory=dict)
    activities_to_refine: List[Dict] = field(default_factory=list)
    iteration_count: int = 0


class TeamProjectState(TypedDict):
    user_request: str
    research_report: str
    skeleton_json: Dict
    enriched_json: Dict
    aligned_json: Dict
    abstracted_json: Dict
    assembled_ir: Dict
    final_ir: Dict
    last_agent_called: str


def run_workflow_logic(user_request, modelscope_key, tavily_key, classifier_path):
    """
    封装整个工作流执行逻辑
    """
    # 1. 设置全局变量
    global GLOBAL_TAVILY_KEY, GLOBAL_CLIENT
    GLOBAL_TAVILY_KEY = tavily_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    GLOBAL_CLIENT = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1',
        api_key=modelscope_key
    )

    # 2. 加载模型
    activity_classifier_pipeline = load_activity_classifier(classifier_path)

    # 3. 初始化 Memory 和 Callback
    memory = SharedMemory()
    token_usage_handler = TokenUsageCallbackHandler()

    # 4. 初始化 LLMs
    llm_base = ChatOpenAI(
        model_name='Qwen/Qwen3-Coder-480B-A35B-Instruct',
        openai_api_base='https://api-inference.modelscope.cn/v1',
        openai_api_key=modelscope_key,
        temperature=0, callbacks=[token_usage_handler]
    )
    llm_reasoning = ChatOpenAI(
        model_name='deepseek-ai/DeepSeek-R1-0528',
        openai_api_base='https://api-inference.modelscope.cn/v1',
        openai_api_key=modelscope_key,
        temperature=0, callbacks=[token_usage_handler], request_timeout=100
    )
    llm_standard = ChatOpenAI(
        model_name='deepseek-ai/DeepSeek-V3.2',
        openai_api_base='https://api-inference.modelscope.cn/v1',
        openai_api_key=modelscope_key,
        temperature=0, callbacks=[token_usage_handler]
    )
    llm_audit = ChatOpenAI(
        model_name='ZhipuAI/GLM-4.7',
        openai_api_base='https://api-inference.modelscope.cn/v1',
        openai_api_key=modelscope_key,
        temperature=0, callbacks=[token_usage_handler]
    )

    # 5. 定义辅助函数 (Extraction, Aligner Logic)
    def perform_extraction(process_description: str, user_req: str) -> str:
        print(f"\n--- [Core Logic] Performing extraction... ---\n")
        prompt = f"""
        You are an expert BPMN 2.0 Process Analyst. Your task is to meticulously analyze a detailed business process description and transform it into a logically coherent, structured JSON object representing the core BPMN components: Roles, Activities, and Gateways. You must act as a strict validator, ensuring the output adheres to all BPMN rules.
            
            Please infer other appropriate parallel, inclusive, and exclusive gateways!
            Please infer other appropriate parallel, inclusive, and exclusive gateways!
            Please infer other appropriate parallel, inclusive, and exclusive gateways!
            **PART 1: YOUR CORE MODELING PRINCIPLES** (You MUST follow these):**
            
            **GOLDEN RULE: MUST INCLUDE USER-SPECIFIED ELEMENTS**
            The user's original request is provided below. You MUST ensure that your final JSON includes activities and gateways that directly correspond to the key elements mentioned in that request. This is your highest priority.
            Furthermore, you MUST NOT extract any roles that were not explicitly mentioned in the Original User Request. Your final `roles` list must be a subset of, or identical to, the roles mentioned by the user. For example, if the user only mentions 'procurement team' and 'supplier', your output `roles` list cannot contain 'finance department'.
            **Original User Request:**
            ---
            {user_req}
            ---
            
            1.  **Distinguish Actions from Outcomes**:
                *   An **Activity** is a task to be performed (e.g., "Supervisor: Review expense report").
                *   A **Gateway** is a question that splits the flow (e.g., "Supervisor: Is the report approved?").
                *   The outcomes of a gateway (e.g., "Approved", "Rejected") are NOT activities themselves; they are paths leading to the *next* activity. Do NOT create activities like "Supervisor: Approve report". The approval is the *result* of the "Review" activity, represented by the gateway's path.
        
            2.  **Ensure Logical Consistency**:
                *   Every path branching from a gateway MUST lead to a distinct, subsequent activity that you have also extracted.
                *   For every decision described in the text, you MUST extract both the gateway (the question) and the activities that result from each choice (e.g., the 'Yes' path and the 'No' path).
        
            3.  **Preserve Maximum Granularity**:
                *   Your primary goal is to extract **every single step** described in the process document as a distinct activity.
                *   **DO NOT merge or simplify activities** at this stage. If the document says "Select supplier" and then "Create Purchase Order", you MUST extract two separate activities.
                *   Extract all steps, even if they seem minor. The goal is a 1:1 conversion of the text description to a structured format. Subsequent steps in the pipeline will handle refinement.
        
        
            **PART 2: FORMAL BPMN GATEWAY RULES (You MUST strictly adhere to these)**
              *   **General Rule**: Every gateway you extract must have at least two branches. A decision point with only one outcome is not a gateway.
        
            *   **Exclusive Gateway (XOR)**:
                *   Represents an "EITHER/OR" decision.
                *   The branches represent mutually exclusive paths.
                *   **Example Logic**: "If the amount is over $500, route to supervisor; otherwise, auto-approve."
        
            *   **Parallel Gateway (AND)**:
                *   Represents concurrent actions ("DO ALL").
                *   All branches are activated simultaneously.
                *   **Example Logic**: "The warehouse packs the items, and at the same time, finance sends the invoice."
        
            *   **Inclusive Gateway (OR)**:
                *   Represents "ONE OR MORE" optional paths.
                *   One, several, or all branches can be activated.
                *   **Example Logic**: "The customer can select optional services: data backup, software installation, or both."
            
            **PART 3: CRITICAL OUTPUT INSTRUCTIONS & JSON SCHEMA**
            
            *   **`roles`**: A `list` of `string`s.
            *   **`activities`**: A `list` of `object`s. Each object has an `id` (`string`) and `description` (`string` in "Role: Action" format).
            *   **`gateways`**: A `list` of `object`s. Each object has an `id` (`string`), `type` (`string`: "exclusiveGateway", "parallelGateway", or "inclusiveGateway"), and `description` (`string` as a question).
            *   **Strict JSON Output**: Your entire output must be a single, valid JSON object.
        
            ---
            **PART 4: HIGH-QUALITY EXAMPLES (Study these to understand the application of the rules)**
            **EXAMPLE 1: Exclusive Gateway (XOR)**
        
            **Input Text:**
            "## Expense Reimbursement Process
            ### Step 1: Submission & Routing
            The process starts when an Employee submits an expense report. The system immediately evaluates the total amount. If the total is over $500, it is routed to the employee's Supervisor for manual review. Otherwise, it is sent for automated compliance checks.
            ### Step 2: Supervisor Review
            The Supervisor reviews the report and makes a decision. If they approve it, the report is forwarded to the Finance team. If they reject it, a notification is sent back to the Employee, who can then correct and resubmit the report."
        
            **Your Correct JSON Output:**
            ```json
            {{
              "roles": ["Employee", "System", "Supervisor", "Finance team"],
              "activities": [
                {{"id": "act_1", "description": "Employee: Submit expense report"}},
                {{"id": "act_2", "description": "System: Perform automated compliance checks"}},
                {{"id": "act_3", "description": "Supervisor: Manually review expense report"}},
                {{"id": "act_4", "description": "Finance team: Process payment"}},
                {{"id": "act_5", "description": "Employee: Correct and resubmit report"}}
              ],
              "gateways": [
                {{"id": "gate_1", "type": "exclusiveGateway", "description": "System: Is the total amount over $500?"}},
                {{"id": "gate_2", "type": "exclusiveGateway", "description": "Supervisor: Is the expense report approved?"}}
              ]
            }}
            ```
            ---
            **EXAMPLE 2: Parallel Gateway (AND)**
        
            **Input Text:**
            "## Order Fulfillment Process
            ### Step 1: Order Confirmation
            A Sales Rep confirms a customer's order in the CRM system. Once confirmed, two processes must start simultaneously.
            ### Step 2: Parallel Processing
            The Warehouse team begins to pick and pack the items for shipment. At the same time, the Finance department generates and sends the invoice to the customer. Both of these actions must be completed before the process can continue.
            ### Step 3: Shipment
            After the items are packed and the invoice is sent, the Logistics team arranges for the shipment of the package.
        
            **Your Correct JSON Output:**
            ```json
            {{
              "roles": ["Sales Rep", "Warehouse team", "Finance department", "Logistics team"],
              "activities": [
                {{"id": "act_1", "description": "Sales Rep: Confirm customer order"}},
                {{"id": "act_2", "description": "Warehouse team: Pick and pack items"}},
                {{"id": "act_3", "description": "Finance department: Generate and send invoice"}},
                {{"id": "act_4", "description": "Logistics team: Arrange for shipment"}}
              ],
              "gateways": [
                {{
                  "id": "gate_1",
                  "type": "parallelGateway",
                  "description": "System: Initiate parallel fulfillment tasks"
                }}
              ]
            }}
            ```
            ---
            **EXAMPLE 3: Inclusive Gateway (OR)**
        
            **Input Text:**
            "## IT Service Request
            ### Step 1: Request Submission
            An Employee submits a request for a new laptop. The IT Support team receives the request.
            ### Step 2: Optional Services Selection
            As part of the setup, the employee can choose one or more additional services. The available options are: a data backup from their old machine, installation of specialized software, and an extended warranty. They can select any combination of these, or none at all.
            ### Step 3: Service Execution
            Based on the employee's selection, the IT Support team performs the requested services before delivering the new laptop.
        
            **Your Correct JSON Output:**
            ```json
            {{
              "roles": ["Employee", "IT Support team"],
              "activities": [
                {{"id": "act_1", "description": "Employee: Submit request for new laptop"}},
                {{"id": "act_2", "description": "IT Support team: Perform data backup"}},
                {{"id": "act_3", "description": "IT Support team: Install specialized software"}},
                {{"id": "act_4", "description": "IT Support team: Purchase extended warranty"}},
                {{"id": "act_5", "description": "IT Support team: Deliver new laptop"}}
              ],
              "gateways": [
                {{
                  "id": "gate_1",
                  "type": "inclusiveGateway",
                  "description": "Employee: Which optional services are selected?"
                }}
              ]
            }}
            ```
            ---
        
            **YOUR CURRENT TASK:**
            Now, as a strict BPMN 2.0 validator and analyst, apply all principles, rules, and learnings from the examples to the following business process description.
            Please note that extracted roles only include roles in the user request!
            Let's think step by step!
        """
        response = GLOBAL_CLIENT.chat.completions.create(
            model='Qwen/Qwen3-Next-80B-A3B-Instruct',
            messages=[{"role": "user", "content": prompt},
                      {"role": "user", "content": f"Here is the business process description:\n\n{process_description}"}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def invoke_with_cost(agent_executor, input_data, name, llm_obj):
        token_usage_handler.get_and_reset_totals()
        print(f"--- Invoking {name}... ---")
        response = agent_executor.invoke(input_data)
        totals = token_usage_handler.get_and_reset_totals()
        cost_info = {"model_name": llm_obj.model_name, "input_tokens": totals["prompt_tokens"],
                     "output_tokens": totals["completion_tokens"]}
        memory.add_entry(name, input_data, response, cost_info)
        return response

    # 6. 构建 Aligner Sub-Graph (粒度对齐)
    def critique_node(state: GraphState) -> dict:
        print("\n--- [ALIGNER] Critiquing ---")
        # 获取当前的 JSON 数据
        current_json = state.processed_json
        raw_activities = current_json.get("activities", [])

        # --- 【修复开始】数据标准化：防止 activities 是字符串列表 ---
        normalized_activities = []
        is_modified = False

        for idx, act in enumerate(raw_activities):
            if isinstance(act, str):
                # 如果是字符串，将其包装成字典
                print(f"  [Warning] Found string activity, normalizing: {act[:20]}...")
                normalized_activities.append({"id": f"act_norm_{idx}", "description": act})
                is_modified = True
            elif isinstance(act, dict):
                # 如果已经是字典，直接保留
                normalized_activities.append(act)
            else:
                # 其他非法类型跳过
                continue

        # 如果数据被修正过，更新 current_json 中的 activities
        if is_modified:
            current_json["activities"] = normalized_activities

        # 使用标准化后的列表进行后续处理
        acts = normalized_activities
        # --- 【修复结束】 ---

        to_refine = []
        for act in acts:
            desc = act.get("description", "")
            # 容错处理：如果 description 为空，跳过
            if not desc:
                continue

            # 提取动作文本（去除角色前缀）
            action_text = desc.split(":", 1)[1].strip() if ":" in desc else desc

            label = "Standard"
            if activity_classifier_pipeline:
                try:
                    # 截断过长的文本防止模型报错
                    res = activity_classifier_pipeline(action_text[:512])
                    label = res[0]['label']
                except Exception as e:
                    print(f"  [Classifier Error] {e}")
                    label = "Standard"

            if label != "Standard":
                print(f"  - Issue: {desc} -> {label}")
                to_refine.append({"activity": act, "granularity": label})

        # 返回结果。如果数据被修正了，同时更新 processed_json 状态，防止下一步报错
        return {
            "activities_to_refine": to_refine,
            "processed_json": current_json
        }

    def refine_node(state: GraphState) -> dict:
        print("\n--- [ALIGNER_NODE] Refining Activities (1-to-1 Standardization) ---")
        current_json = state.processed_json
        activities_to_refine = state.activities_to_refine
        all_activities = current_json.get("activities", [])

        # 提取描述用于集合运算
        problematic_descriptions = {item['activity']['description'] for item in activities_to_refine}

        # 分离标准活动和问题活动
        standard_activities = [act for act in all_activities if act['description'] not in problematic_descriptions]
        problematic_activities = [item['activity'] for item in activities_to_refine]

        print(f"  - Found {len(standard_activities)} standard activities (Keep as is).")
        print(f"  - Found {len(problematic_activities)} problematic activities (To be renamed/standardized).")

        if not problematic_activities:
            print("  - No problematic activities found. Skipping refinement.")
            return {
                "processed_json": current_json,
                "iteration_count": state.iteration_count + 1
            }

        # 生成详细的错误报告，包含具体的粒度问题 (Too Fine / Too Coarse)
        # 格式: "- [Too Fine] 'client: Access website'"
        error_report_lines = []
        for item in activities_to_refine:
            issue_type = item['granularity']  # "Too Fine" or "Too Coarse"
            desc = item['activity']['description']
            error_report_lines.append(f"- [{issue_type}] Activity: '{desc}'")
        error_report_str = "\n".join(error_report_lines)

        standard_activity_examples = """
                "Receive customer inquiry",
                "Address customer concerns",
                "Collect customer information",
                "Provide quote",
                "Place order",
                "Record order in system",
                "Send order confirmation",
                "Conduct initial phone interviews",
                "Check references",
                "Extend offer",
                "Negotiate salary",
                "Check current inventory level",
                "Place order with suppliers",
                "Receive stock",
                "Inspect stock for quality",
                "Conduct site visit",
                "Select supplier",
                "Begin contract negotiations",
                "Sign contract",
                "Onboard supplier",
                "Execute contract",
                "Propose corrective actions",
                "Implement fix",
                "Change policy",
                "Conduct training",
                "Conduct follow-up",
                "Close incident report",
                "Notify all stakeholders",
                "Identify idea for new product or improvement",
                """.strip()

        # --- 核心修改：Prompt 逻辑重构 ---
        refinement_prompt = f"""
                You are an expert BPMN Process Refiner. Your goal is to standardize the descriptions of specific activities to satisfy a strict granularity classifier.

                ### THE STRATEGY: 1-TO-1 REPLACEMENT
                **DO NOT SPLIT activities.** (Splitting causes explosion).
                **DO NOT MERGE activities.** (Merging causes confusion).

                **Your Task:** Take each problematic activity and **RENAME / REPHRASE** it into exactly **ONE** "Standard" operational activity.

                ### INSTRUCTIONS BY TYPE

                1. **IF FLAGGED "TOO COARSE" (Too Vague/Abstract):**
                   - **Problem:** The description sounds like a phase or a goal (e.g., "Manage Recruitment").
                   - **Fix:** Rename it to the *primary concrete action* represented by that step.
                   - **Example:** "HR: Manage Recruitment" -> "HR: Execute recruitment workflow".
                   - **Example:** "Department: Define Responsibilities" -> "Department: Document core role duties".

                2. **IF FLAGGED "TOO FINE" (Too Detailed/Mechanical):**
                   - **Problem:** The description sounds like a keystroke or micro-step (e.g., "Click Submit", "Type Name").
                   - **Fix:** Rename it to the *business-level task* it represents.
                   - **Example:** "Candidate: Type Name" -> "Candidate: Enter personal details".
                   - **Example:** "System: Click Save" -> "System: Record data entry".

                3. **THE "JUDGE" RULE (Crucial):**
                   - The automated classifier is overly sensitive. 
                   - If an activity looks reasonable (e.g., "Schedule interview"), **DO NOT CHANGE IT**. Just output it exactly as it was.
                4. **ANALYZE THE ERROR REPORT**: Look at the list below to see WHY each activity is wrong.
                5. **PRESERVE CONTEXT**: Keep the "Role: Action" format (e.g., "client: Submit form").
                6. **INTEGRATE**: Combine your *newly created* activities with the *unchanged* standard activities.
                ---
                ### PART 1: ERROR REPORT (TARGETS FOR MODIFICATION)
                **Do NOT output these descriptions as they are. You MUST change them.**
                {error_report_str}

                ---
                ### PART 2: STANDARD ACTIVITIES (KEEP THESE EXACTLY AS IS)
                {json.dumps(standard_activities, indent=2)}

                ---
                ### PART 3: REFERENCE CONTEXT (Original Flow)
                Use this only to understand the sequence. **DO NOT COPY THE STRUCTURE BLINDLY.**
                {json.dumps(current_json, indent=2)}

                ---
                ### EXAMPLES OF DESIRED GRANULARITY
                {standard_activity_examples}

                ---
                ### OUTPUT FORMAT
                1. **Transformation Plan:** Briefly list the changes (e.g., "Old Name" -> "New Name").
                2. **Final JSON:** A single valid JSON object containing `roles`, `activities`, and `gateways`.


                **Final JSON Structure:**
                ```json
                {{
                  "roles": [...],
                  "activities": [
                    {{ "id": "...", "description": "..." }},
                    ...
                  ],
                  "gateways": [...]
                }}
                ```
                """

        messages = [
            {"role": "system", "content": refinement_prompt},
            {"role": "user",
             "content": "Please fix the granularity issues based on the Error Report. Output the Transformation Plan first, then the JSON."}
        ]

        # 建议稍微调高一点 temperature，让模型有“创造性”去合并/拆分，而不是死板地复制
        response = GLOBAL_CLIENT.chat.completions.create(
            model='Qwen/Qwen3-235B-A22B-Instruct-2507',
            # Qwen/Qwen3-235B-A22B-Instruct-2507  Qwen/Qwen3-Coder-480B-A35B-Instruct
            messages=messages,
            temperature=0,  # 稍微增加一点随机性，避免死循环复制
            stream=False
        )
        llm_output = response.choices[0].message.content
        print(f"LLM refinement output:\n{llm_output}")

        # --- 增强的 JSON 提取逻辑 ---
        try:
            # 优先寻找 Markdown 代码块
            code_block_pattern = r"```json\s*(\{.*?\})\s*```"
            matches = re.findall(code_block_pattern, llm_output, re.DOTALL)

            if matches:
                # 取最后一个代码块（通常是最终结果）
                json_string = matches[-1]
            else:
                # 如果没有代码块，尝试寻找最外层的大括号
                match = re.search(r'\{.*}', llm_output, re.DOTALL)
                if match:
                    json_string = match.group(0)
                else:
                    raise ValueError("No JSON object found in the LLM output.")

            refined_json = json.loads(json_string)

            # 简单的校验：如果活动数量完全没变，且描述完全没变，可能需要警告
            # (这里可以加额外的逻辑，但先让它跑起来)

            return {
                "processed_json": refined_json,
                "iteration_count": state.iteration_count + 1
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"!!! [ERROR] Failed to parse JSON from LLM response. Error: {e} !!!")
            # 打印出错的片段方便调试
            print(f"Debug - Failed JSON string snippet: {llm_output[-500:]}")
            return {
                "processed_json": state.processed_json,
                "iteration_count": state.iteration_count + 1
            }

    def should_continue(state: GraphState):
        print("\n--- [ALIGNER_EDGE] Deciding to continue or finish ---")
        if not state.activities_to_refine:
            print("  - Decision: All activities are standard. FINISH.")
            return "end"
        elif state.iteration_count >= 8:
            print("  - Decision: Max iterations reached. FINISH.")
            return "end"
        else:
            print(f"  - Decision: Found {len(state.activities_to_refine)} issues. CONTINUE to refine.")
            return "continue"

    aligner_workflow = StateGraph(cast(Any, GraphState))
    aligner_workflow.add_node("critique", critique_node)
    aligner_workflow.add_node("refine", refine_node)
    aligner_workflow.set_entry_point("critique")
    aligner_workflow.add_conditional_edges("critique", should_continue, {"continue": "refine", "end": END})
    aligner_workflow.add_edge("refine", "critique")
    aligner_app = aligner_workflow.compile()

    # 重新定义 align_activity_granularity 工具以使用本地构建的 app
    @tool(return_direct=True)
    def local_align_tool(input_data: Union[str, dict]) -> str:
        """Local wrapper for alignment."""
        print("\n--- [ALIGNER TOOL] Starting Loop ---")
        inp = json.loads(input_data) if isinstance(input_data, str) else input_data
        final = aligner_app.invoke(
            {"input_json": inp, "processed_json": inp, "activities_to_refine": [], "iteration_count": 0})
        return json.dumps(final["processed_json"], indent=2, ensure_ascii=False)

    # 7. 构建 Agents
    prompt_hub = hub.pull("hwchase17/react")

    # Researcher
    instruction_researcher = """
    You are a Senior Process Researcher. Your mission is to transform a user's brief, coarse-grained request into a comprehensive, detailed, and professional business process design document.

    **SHARED MEMORY LOG (Previous Steps):**
    ---
    {memory_log}
    ---

    **YOUR METHODOLOGY (You MUST follow this three-stage process):**

        **STAGE 1: DECONSTRUCTION & DYNAMIC QUERY FORMULATION (Internal Thought)**
        - First, meticulously analyze the user's request. Deconstruct it to identify all the key entities provided. These include:
            1.  The **Core Process Name** (e.g., "hardware procurement").
            2.  All mandatory **Roles** (e.g., "IT department", "employee").
            3.  All key **Activities** (e.g., "analyse the request").
            4.  All key **Decision Points** (e.g., "management approval").
        - Next, **dynamically construct a single, comprehensive search query** by weaving together all the keywords you just identified.
        - Your goal is to create a query that is highly specific to the user's request to get the most relevant results about the entire process lifecycle, including best practices and common failure points (unhappy paths).

        **STAGE 2: RESEARCH EXECUTION (MANDATORY: ONE SEARCH ONLY)**
        - You MUST execute your single, dynamically constructed query by calling the `tavily_wrapper` tool **only once**.
         - After this single action, you are **STRICTLY FORBIDDEN** from calling `tavily_wrapper` again, regardless of the quality of the results. Your task is to work with the information you receive.

        **STAGE 3: SYNTHESIS & DRAFTING (Final Answer)**
        - **Immediately after observing the single search result**, you MUST synthesize a final report.
        - **If the search results are sufficient:** Build your report around the user's mandatory roles and activities, fleshed out with the details and best practices you discovered.
        - **If the search results are insufficient or too high-level:** DO NOT SEARCH AGAIN. Instead, synthesize a "best-effort" process design based on the limited information you have gathered AND your own extensive internal knowledge of business processes. In this case, you **MUST** begin your final report with a disclaimer, for example: "Note: The following process is a best-effort design based on high-level industry principles, as detailed public sources were not available."
        - **After you synthesize a final report,your very next action MUST be to call the `terminate` tool,pass a final report from `tavily_wrapper` tool to the `terminate` tool.
        - The `Action Input` for the `terminate` tool MUST be final report. - The input for the `terminate` tool MUST be the **complete design document you just synthesized in your thought process**, NOT the raw output from the search tool.

        **MUST**:You must call once the `tavily_wrapper` tool, no matter how good the quality of the result is.

    ---
        **EXAMPLE of your Thought Process (Based on a specific user request):**

        **User Request:** "Please design a hardware procurement process that includes the roles of 'IT department', 'employee', 'supplier', 'management', and 'financial department'. The key activities are 'analyse the request' and 'prepare an order', and there should be decision points regarding 'IT department approval' and 'management approval'."

        **Your Correct Thought Process:**
        Thought: The user wants a 'hardware procurement process'. I must first deconstruct their request to find all the specific keywords they provided.
        - Core Process: hardware procurement
        - Key Roles: 'IT department', 'employee', 'supplier', 'management', 'financial department'
        - Key Activities & Decisions: 'analyse the request', 'prepare an order', 'IT department approval', 'management approval'
        Now, I will combine all these specific keywords into a single, powerful search query to find a complete workflow that covers everything from request to fulfillment, including best practices and common issues.
        Action: tavily_wrapper
        Action Input: "detailed hardware procurement workflow steps involving employee request, IT department analysis, management approval, supplier order, and financial department processing best practices and common issues"
        Observation: [a final report]
        Thought: I have received the research results. According to my instructions, my next and final action is to call the `terminate` tool with these results.
        Action: terminate
        Action Input: [a final report]
        .......

        Let's think step by step！
    """
    researcher_template = instruction_researcher + "\n\n" + prompt_hub.template
    res_prompt = PromptTemplate.from_template(researcher_template)
    res_agent = create_react_agent(llm_base, [tavily_wrapper, terminate], res_prompt)
    res_executor = AgentExecutor(agent=res_agent, tools=[tavily_wrapper, terminate], verbose=True,
                                 handle_parsing_errors=True)

    # Enricher
    instruction_gateway = """
    You are a "Gateway Inference Specialist" Agent. Your sole purpose is to take a preliminary, "skeleton" process and intelligently enrich it by inserting the necessary logical gateways and their corresponding new activities. Your mission has two parts: first, to complete existing gateways, and second, to actively infer and create new ones.


    You MUST base your inferences on the **"Original Research Report"** and CONTEXTUAL INFORMATION provided below. This document contains the detailed business logic, exception paths, and decision points. The "Skeleton JSON" is just a starting point; the Research Report is your guide for enriching it.
    Please infer other appropriate parallel, inclusive, and exclusive gateways!
    ---
    **CONTEXTUAL INFORMATION:**

    **1. Original User Request:**
    {user_request}

    **2. Original Research Report:**
    {research_report}

    **3. Skeleton JSON to Enrich:**
    {skeleton_json}

    **4. SHARED MEMORY LOG (Previous Steps):**
    {memory_log}
    ---


    YOUR NON-NEGOTIABLE PRINCIPLE: The "One-to-Many" Mapping Rule

    For every single gateway you add or complete, you MUST ALSO add brand new activities—one for each logical branch that the gateway creates. There are no exceptions.

    PART 1: COMPLETING EXISTING GATEWAYS - The "Complete Scenario" Mandate

    You are not just adding symbols; you are modeling complete business scenarios. For every gateway already present in the skeleton JSON, you MUST follow this mandatory workflow:

    Identify the Decision Point: Locate an existing gateway (e.g., "Management Approval").

    Analyze the Gateway Question: Understand the decision being made (e.g., "Is the purchase approved by Management?").

    Brainstorm ALL Plausible Business Outcomes: For the decision, think like an experienced analyst. What are ALL the possible results?

    For an approval, the outcomes are almost always: 'Approved', 'Rejected', and often 'Needs Rework'.

    Create a NEW, DEDICATED Activity for EACH MISSING Outcome: This is a critical step. If an existing gateway only has a path for 'Approved', you MUST create new act_infer_... activities for the 'Rejected' and 'Rework' branches.

    Example of Your Thought Process for COMPLETION:
    Thought: The skeleton JSON has a gateway gate_2: "Management: Is the purchase necessary and within budget?". This is an approval decision. I must ensure all plausible outcomes are modeled. The 'Yes' path (Approved) is already connected. I MUST now add new activities for the 'No' (Rejected) and 'Rework' paths to make the process logically complete.
    I will add:

    act_infer_1: "Management: Reject request and notify employee"

    act_infer_2: "Management: Send request back for budget adjustment"
    This fulfills the "Complete Scenario" Mandate.

    PART 2: ACTIVELY INFERRING NEW GATEWAYS - Your Role as a Business Analyst

    After ensuring all existing gateways are complete, your next and most critical task is to actively infer new gateways to make the process more robust and efficient.

    1. INFERRING Exclusive Gateways (XOR) - The Decision Points
    WHEN TO INFER: Look for approval steps, quality checks, or validation activities in the skeleton that DON'T have a gateway after them. An activity like "Department: Review request" almost always implies a subsequent decision.
    YOUR ACTION: If you see a review activity, you MUST INFER and add a new exclusiveGateway immediately after it, along with new activities for all its outcomes (e.g., 'Approved', 'Rejected').
    Please do not infer meaningless gateways!
    2. INFERRING Parallel Gateways (AND) - The Concurrent Tasks
    WHEN TO INFER: Look for activities performed by different roles that do not strictly depend on each other. This is a key opportunity for efficiency.
    YOUR ACTION: If you see two sequential activities like "Warehouse: Pack items" followed by "Finance: Send invoice", you SHOULD INFER that these can happen in parallel. Insert a parallelGateway before them and model them as concurrent paths with new corresponding activities.
    Please do not infer meaningless gateways!
    3. INFERRING Inclusive Gateways (OR) - The Optional Tasks
    WHEN TO INFER: Look for activities that sound optional or conditional. Read the original user request for clues about optional steps.
    YOUR ACTION: If the process mentions optional services (e.g., "data migration," "extended warranty"), you should infer an inclusiveGateway to model these choices, along with new activities for the options.
    Please do not infer meaningless gateways!


    CRITICAL EXECUTION RULES (Apply to ALL actions):
    Preserve the Core: You MUST NOT remove or alter original elements. Your job is to ADD and ENRICH.
    You are a Flow Splitter, NOT a Merger: You are STRICTLY FORBIDDEN from adding 'merging' gateways.
    THE CARDINAL RULE: One Activity Per Branch, No Exceptions.
    exclusiveGateway -> >= 2 new activities.
    parallelGateway -> >= 2 new activities.
    inclusiveGateway -> >= 1 new activity.
    MANDATORY SELF-CORRECTION AUDIT:
    Before providing your final JSON, you MUST perform this final audit on your own work:

    For each exclusiveGateway I added or completed, did I also add AT LEAST TWO new activities corresponding to its branches? (Yes/No)

    For each parallelGateway I added, did I also add AT LEAST TWO new activities corresponding to its branches? (Yes/No)
    If the answer to any of these questions is NO, you have failed your primary directive and you MUST go back and add the missing activities before finalizing your answer.

    YOUR TASK:

    Analyze the provided Skeleton JSON and Original Request.

    1、First, apply the "Complete Scenario" Mandate to all gateways already present in the skeleton.
    2、Second, apply your Business Analyst skills to actively infer and add NEW gateways of all types (exclusive, parallel, inclusive) where logically appropriate.
    3、Ensure every action adheres to all Critical Execution Rules.
    4、Provide the complete, enriched JSON that has passed your self-correction audit.
    5、Only infer meaningful gateways. After inferring a gateway, it is necessary to infer its branch activities.
    6、Please infer other appropriate parallel, inclusive, and exclusive gateways,and ensure that only infer meaningful gateways.
    FINAL OUTPUT INSTRUCTION:
    After you have completed the enrichment and self-audit, your work is done.
    Your final thought MUST be "I have completed the enrichment by both completing existing gateways and inferring new ones.
    I will now call the finalize_process_design tool."
    You MUST then call the finalize_process_design tool with the complete JSON string as the input.
    Let's think step by step！
    """
    gateway_template = instruction_gateway + "\n\n" + prompt_hub.template
    enrich_prompt = PromptTemplate.from_template(gateway_template)
    enrich_agent = create_react_agent(llm_reasoning, [finalize_process_design], enrich_prompt)
    enrich_executor = AgentExecutor(agent=enrich_agent, tools=[finalize_process_design], verbose=True,
                                    handle_parsing_errors=True)

    # Aligner Agent (Dispatcher)
    instruction_aligner = """
    You are a simple, rule-based dispatcher. Your ONLY function is to take the user's input and immediately call the align_activity_granularity tool with that exact input.

    **SHARED MEMORY LOG (Previous Steps):**
    ---
    {memory_log}
    ---

    CRITICAL RULE: You MUST follow the Thought/Action/Action Input format.
    Your thought process should be extremely simple. After thinking, you MUST immediately output an 'Action' and 'Action Input'.

    EXAMPLE of your ONLY valid response format:

    Thought: The user has provided a JSON string. My only job is to call the align_activity_granularity tool with this string.
    Action: align_activity_granularity
    Action Input: [the complete JSON string provided by the user]
    """
    aligner_template = instruction_aligner + "\n\n" + prompt_hub.template
    align_prompt = PromptTemplate.from_template(aligner_template)
    align_agent = create_react_agent(llm_standard, [local_align_tool], align_prompt)
    align_executor = AgentExecutor(agent=align_agent, tools=[local_align_tool], verbose=True,
                                   handle_parsing_errors=True)

    # Pruner
    instruction_pruning = """
    You are a "BPMN Quality Assurance & Standardization Analyst" Agent. Your final and most critical mission is to take a detailed, enriched process JSON and perform a final quality check to ensure its granularity and logic match the highest operational standards.
    You will be given a "Detailed JSON to Refine" (enclosed in <JSON_START> and <JSON_END> tags) and the "Original User Request".
    Your first step is to extract the JSON content from between these tags.
    SHARED MEMORY LOG (Previous Steps):
    {memory_log}
    YOUR METHODOLOGY: "Final Polish & Standardization"
    PRINCIPLE 1: THE GRANULARITY GOLD STANDARD (Your North Star)
    Your absolute priority is to ensure the final activities have an operational level of detail. You have been shown examples of high-quality process models. Your output MUST match that level of granularity.
    Reference Standard: A good activity describes a single, clear action by a specific role. Examples of PERFECT granularity are: "IT department prepares an order and sends this to the supplier", "The financial department finds resources", "The management department decides whether to approve the request".
    Anti-Pattern to Avoid: Do NOT merge distinct operational steps. For example, "Analyse request" and "Prepare order" are separate, value-adding activities and MUST remain separate.
    NEW: Gateway Granularity Standard (Your Pruning Guide for Gateways)
    Prune Gateways that are TOO FINE-GRAINED (Redundant Micro-decisions): If an activity's description already implies a check (e.g., "IT Department: Verify PO accuracy"), a subsequent gateway asking "Is the PO accurate?" is redundant and MUST be pruned. The outcome of the verification activity should directly lead to the next steps.
    Prune Gateways that are TOO COARSE-GRAINED (Vague Questions): A gateway with a generic question like "System: Proceed?" or "Is everything okay?" lacks business value and MUST be pruned. Decision points must be specific and meaningful, like "Management: Is the budget approved?".
    Please refer agent enricher's expansion reason,do not arbitrarily delete meaningful gateways.
    PRINCIPLE 2:FAVOR SIMPLICITY AND THE HAPPY PATH
    Your Goal is Clarity: The final model should be easy for a business user to understand. Avoid modeling every single exception path.
    Prune unnecessary Unhappy Paths: Unless an exception path is critical (e.g., a legally required compliance check), you should prune most simple rejection or rework activities.
    PRINCIPLE 3: PRESERVE ALL LOGICAL PATHS (Do No Harm)
    Your default action is to KEEP everything. You are not a simplifier; you are a validator.
    GOLDEN RULE: You are FORBIDDEN from pruning any key activities or decision points from the "Original User Request".
    EXCEPTION RULE FOR PRUNING: You should only prune an activity if it is a pure, low-value notification AND its removal does not break a logical flow from a gateway. For example, "Notify employee of rejection" can be pruned IF the "rejection" branch from the gateway is re-routed to a more meaningful step like "End Process" or "Rework Request".
    PRINCIPLE 4: THE UNBREAKABLE LINK AUDIT (Your Final, Most Important Duty)
    After any cleaning or merging, your final task is to act as a strict auditor of logical flow.
    MANDATORY AUDIT PROCEDURE: For every single gateway remaining in the JSON, you MUST perform the following check:
    Identify the Gateway: Look at its description (e.g., "Is the request approved?").
    Identify its Logical Branches: Based on the question, determine the implied branches (e.g., a 'Yes' branch and a 'No' branch).
    Validate Each Branch's Destination: For EACH branch, you must confirm that it logically connects to an activity that still exists in your refined activities list.
    CORRECTION MANDATE: If you discover a "dangling branch" (a branch that points to nowhere because its target activity was pruned), you have FAILED the audit. You MUST correct it. Your only option is to ADD a logical concluding activity for that branch.
    Example: If the 'No' branch of "Is the request approved?" is dangling, you MUST add a new, simple, concluding activity like {{"id": "act_end_1", "description": "System: Close request as rejected"}} to terminate that path correctly.
    A process model with incomplete gateway logic is invalid. This audit is non-negotiable.
    YOUR TASK:
    Review the "Detailed JSON" against the "Granularity Gold Standard". Identify and lock the core elements from the "Original User Request". Identify any activities that are too broad or too granular. Perform minimal, necessary merges (e.g., "Send verbal offer" + "Send written offer" -> "Extend formal offer").
    Identify and prune only the lowest-value notification/logging activities, as per Principle 2.
    Perform the final, critical Logical Flow Mandate check. Go through every gateway and ensure all its paths are correctly connected. This is the most important step.Perform the "Unbreakable Link Audit" on every gateway. This is your final and most critical validation step. Add concluding activities if necessary to fix any broken links.
    Produce the final, standardized JSON, ensuring it is 100% logically coherent.
    FINAL OUTPUT INSTRUCTION:
    After completing your final quality assurance pass, your work is done.
    Your final thought MUST be "I have validated the process against the gold standard for granularity and logic. I will now call the deliver_abstracted_design tool."
    You MUST then call the deliver_abstracted_design tool with the complete, standardized JSON string as the input.
    Let's think step by step！
    """
    pruning_template = instruction_pruning + "\n\n" + prompt_hub.template
    prune_prompt = PromptTemplate.from_template(pruning_template)
    prune_agent = create_react_agent(llm_audit, [deliver_abstracted_design], prune_prompt)
    prune_executor = AgentExecutor(agent=prune_agent, tools=[deliver_abstracted_design], verbose=True,
                                   handle_parsing_errors=True)

    # Assembler Agent
    instruction_assembly = """
    You are a Process Assembly Orchestrator. Your only job is to take the final, cleaned JSON of process components and pass it to the assemble_process_components tool.
    You must not modify the input. Your final thought must be to call the tool with the received JSON.

    **SHARED MEMORY LOG (Previous Steps):**
    ---
    {memory_log}
    ---

    Let's think step by step！
    """
    assembly_template = instruction_assembly + "\n\n" + prompt_hub.template
    asm_prompt = PromptTemplate.from_template(assembly_template)
    asm_agent = create_react_agent(llm_standard, [assemble_process_components], asm_prompt)
    asm_executor = AgentExecutor(agent=asm_agent, tools=[assemble_process_components], verbose=True,
                                 handle_parsing_errors=True)

    # Auditor
    instruction_auditor = """
    You are a world-class Senior Process Architect, acting as the Final Auditor for a generated business process model.
    Your task is to perform a comprehensive audit on the final structured process Intermediate Representation (IR) to ensure it is logically sound, efficient, complete, and perfectly aligned with the user's original goal.

    You will be given the assembled IR and the original user request, separated by "<<USER_GOAL_SEPARATOR>>".

    **SHARED MEMORY LOG (Previous Steps):**
    ---
    {memory_log}
    ---

    YOUR MULTI-DIMENSIONAL AUDIT PROCESS (You MUST check all of them in your thought process):

    1. Goal Alignment Audit (Strategic Check):
    Objective: Ensure the process completely fulfills the user's core request.
    Action: Compare the process IR against the "Original User Request".
    Checklist:
        -Does the process include all specified roles (e.g., 'department', 'HR', 'candidate')?
        -Are all key activities mentioned by the user (e.g., 'candidate screening', 'offer negotiation') clearly represented?
        -Are all key decision points (e.g., 'candidate selection', 'offer acceptance') present as gateways in the IR?

    2. Core Logic Audit (Sanity Check):
    Objective: Verify the fundamental business logic of the assembled flow.
    Action: Trace the sequences within the IR.
    Checklist:
        -Sequence Errors: Is the order of activities logical? (e.g., "Conduct background check" must happen before "Send finalized employment contract").
        -Gateway Logic Errors: Are gateway types correct? (e.g., a candidate cannot both 'Accept Offer' and 'Decline Offer' in parallel; this must be an exclusive choice).
        -Misplaced Activities: Is any activity in the wrong place? (e.g., "Extend Offer" should be inside the "Candidate Selected" branch, not before it).

    3. Efficiency & Best Practice Audit (Optimization Check):
    Objective: Identify opportunities to make the process smarter and more efficient.
    Action: Look for anti-patterns or optimization opportunities.
    Checklist:
        -Parallelism: Are there sequential activities that could run in parallel? (e.g., "HR: Conduct background checks" and "Department: Prepare onboarding plan"). If so, suggest wrapping them in a parallelGateway.
        -Redundancy: Are there duplicate or unnecessary steps? (e.g., two separate activities for "Verify candidate details" by the same role).Do not prun any gateway!
        -Role Sanity: Is the role assigned to each task appropriate for that task? (e.g., a Candidate should not be performing an internal HR task).

    4. Completeness & Boundary Audit (Structural Integrity Check):
    Objective: Ensure the process is a well-formed, complete graph with no dead ends.
    Action: Traverse every branch of every gateway in the IR.
    Checklist:
        -Does every single sequence within every branch lead to at least one subsequent activity or gateway?
        -Are there any empty sequence arrays ([])? This is a critical error that must be fixed, often by adding a concluding activity.

    FEW-SHOT EXAMPLES OF AUDITING AND CORRECTION:
    --- EXAMPLE 1: Correcting a Logic Error and a Completeness Error ---
    Process IR to be Audited:
    {
      "process": [
        { "type": "activity", "id": "act_1", "description": "Manager: Review document" },
        { "type": "activity", "id": "act_2", "description": "User: Submit document" },
        {
          "type": "exclusiveGateway", "id": "gate_1", "description": "Is document approved?",
          "branches": [
            { "condition": "Yes", "sequence": [ { "type": "activity", "id": "act_3", "description": "System: Publish document" } ] },
            { "condition": "No", "sequence": [] }
          ]
        }
      ]
    }

    Your Thought Process:
    1、Goal Alignment Audit: All key activities (submit, review, publish) and the decision point (approval) are present. PASS.
    2、Core Logic Audit: There is a major Sequence Error. "Manager: Review document" (act_1) appears before "User: Submit document" (act_2). This is impossible. The submission must come first. NEEDS REVISION.
    3、Efficiency Audit: The process is simple, no major efficiency issues. PASS.
    4、Completeness Audit: The "No" branch of gate_1 has an empty sequence ([]). This is a critical Dead End. The user request specifies that the user should be notified. NEEDS REVISION.

    Decision: Needs Revision.

    Corrected IR JSON:
    {
      "process": [
        { "type": "activity", "id": "act_2", "description": "User: Submit document" },
        { "type": "activity", "id": "act_1", "description": "Manager: Review document" },
        {
          "type": "exclusiveGateway", "id": "gate_1", "description": "Is document approved?",
          "branches": [
            { "condition": "Yes", "sequence": [ { "type": "activity", "id": "act_3", "description": "System: Publish document" } ] },
            { "condition": "No", "sequence": [ { "type": "activity", "id": "act_4_new", "description": "System: Notify user of rejection" } ] }
          ]
        }
      ]
    }

    --- EXAMPLE 2: Correcting an Efficiency (Parallelism) Error ---
    Process IR to be Audited:
    {
    "process": [
    { "type": "activity", "id": "act_1", "description": "Sales: Confirm order" },
    { "type": "activity", "id": "act_2", "description": "Warehouse: Pack items" },
    { "type": "activity", "id": "act_3", "description": "Finance: Send invoice" },
    { "type": "activity", "id": "act_4", "description": "Logistics: Ship order" }
    ]
    }

    Your Thought Process:
    1、Goal Alignment Audit: All key activities are present. PASS.

    2、Core Logic Audit: The sequence is logical. PASS.

    3、Efficiency & Best Practice Audit: The user request explicitly states that packing and invoicing happen "at the same time". The current IR shows them as sequential (act_2 then act_3). This is a clear opportunity for Parallelism. NEEDS REVISION.

    4、Completeness Audit: No gateways, so no dead ends. PASS.

    Decision: Needs Revision.

    Corrected IR JSON:
    {
      "process": [
        { "type": "activity", "id": "act_1", "description": "Sales: Confirm order" },
        {
          "type": "parallelGateway", "id": "gate_parallel_start",
          "branches": [
            { "sequence": [ { "type": "activity", "id": "act_2", "description": "Warehouse: Pack items" } ] },
            { "sequence": [ { "type": "activity", "id": "act_3", "description": "Finance: Send invoice" } ] }
          ]
        },
        { "type": "activity", "id": "act_4", "description": "Logistics: Ship order" }
      ]
    }

    YOUR TASK:
    1、First, in your thought process, go through each of the four audit dimensions one by one. State your findings for each dimension clearly.
    2、Based on your complete audit, decide if the process IR "Is Approved" or "Needs Revision".
    3、If it "Needs Revision", you MUST generate a corrected version of the entire IR JSON that fixes ALL the issues you identified.
    4、If it "Is Approved", you will use the original, unchanged IR JSON.
    5、Your final action MUST be to call the finalize_audit tool, passing the final (either corrected or approved) IR JSON string as the input.
    6、Your final output MUST be a single, perfectly valid JSON object. Before you finish, mentally trace every bracket, brace, and comma to ensure the syntax is flawless. An invalid JSON output is a complete failure of your task.
    Let's think step by step！
    """
    escaped_instruction_auditor = instruction_auditor.replace("{", "{{").replace("}", "}}")
    full_auditor_template = escaped_instruction_auditor + "\n\n" + prompt_hub.template
    audit_prompt = PromptTemplate.from_template(full_auditor_template)
    audit_agent = create_react_agent(llm_audit, [finalize_audit], audit_prompt)
    audit_executor = AgentExecutor(agent=audit_agent, tools=[finalize_audit], verbose=True, handle_parsing_errors=True)

    # 8. 定义主图节点
    def research_node(state: TeamProjectState):
        inp = {"input": state["user_request"], "memory_log": memory.get_formatted_log()}
        res = invoke_with_cost(res_executor, inp, "Researcher", llm_base)
        return {"research_report": res['output'], "last_agent_called": "Researcher"}

    def extraction_node(state: TeamProjectState):
        json_str = perform_extraction(state["research_report"], state["user_request"])
        memory.add_entry("Extractor", "Extraction", json_str)
        try:
            return {"skeleton_json": json.loads(json_str), "last_agent_called": "Extractor"}
        except:
            return {"skeleton_json": {}, "last_agent_called": "Extractor"}

    def enrichment_node(state: TeamProjectState):
        inp = {
            "input": "Enrich skeleton.",
            "user_request": state["user_request"],
            "research_report": state["research_report"],
            "skeleton_json": json.dumps(state["skeleton_json"]),
            "memory_log": memory.get_formatted_log()
        }
        res = invoke_with_cost(enrich_executor, inp, "Enricher", llm_reasoning)
        return {"enriched_json": json.loads(res['output']), "last_agent_called": "Enricher"}

    def alignment_node(state: TeamProjectState):
        inp = {"input": json.dumps(state["enriched_json"]), "memory_log": memory.get_formatted_log()}
        res = invoke_with_cost(align_executor, inp, "Aligner", llm_standard)
        return {"aligned_json": json.loads(res['output']), "last_agent_called": "Aligner"}

    def pruning_node(state: TeamProjectState):
        inp_str = f"Request: {state['user_request']}\nJSON: {json.dumps(state['aligned_json'])}"
        inp = {"input": inp_str, "memory_log": memory.get_formatted_log()}
        res = invoke_with_cost(prune_executor, inp, "Pruner", llm_audit)
        return {"abstracted_json": json.loads(res['output']), "last_agent_called": "Pruner"}

    def assembly_node(state: TeamProjectState):
        # 准备输入
        # 增加容错：如果 abstracted_json 是 None，给一个空字典
        abstracted_data = state.get("abstracted_json") or {}
        inp = {"input": json.dumps(abstracted_data), "memory_log": memory.get_formatted_log()}

        # 调用 Agent
        res = invoke_with_cost(asm_executor, inp, "Assembler", llm_standard)

        # --- 【修复开始】增加 JSON 解析的容错处理 ---
        output_str = res.get('output', '')

        # 打印调试信息，方便在网页日志看看到底返回了什么
        print(f"--- [Assembler Raw Output] ---\n{str(output_str)[:200]}...\n------------------------------")

        try:
            # 尝试解析 JSON
            assembled_ir = json.loads(output_str)
        except json.JSONDecodeError:
            # 如果解析失败（比如返回了报错字符串，或者空字符串）
            print(f"!!! [Assembler Error] Failed to parse JSON. Raw output: {output_str}")
            # 构造一个包含错误信息的对象，防止流程崩溃
            assembled_ir = {
                "process": [],
                "error": "Assembler failed to produce valid JSON",
                "raw_output": output_str
            }
        except TypeError:
            # 处理 output_str 不是字符串的情况
            print(f"!!! [Assembler Error] Output is not a string: {type(output_str)}")
            assembled_ir = {"process": [], "error": "Invalid output type"}

        return {"assembled_ir": assembled_ir, "last_agent_called": "Assembler"}
        # --- 【修复结束】 ---

    def audit_node(state: TeamProjectState):
        inp_str = f"Request: {state['user_request']}\nIR: {json.dumps(state['assembled_ir'])}"
        inp = {"input": inp_str, "memory_log": memory.get_formatted_log()}
        res = invoke_with_cost(audit_executor, inp, "Auditor", llm_audit)
        return {"final_ir": json.loads(res['output']), "last_agent_called": "Auditor"}

    # 9. 构建主图
    workflow = StateGraph(TeamProjectState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("enricher", enrichment_node)
    workflow.add_node("aligner", alignment_node)
    workflow.add_node("pruner", pruning_node)
    workflow.add_node("assembler", assembly_node)
    workflow.add_node("auditor", audit_node)

    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "extractor")
    workflow.add_edge("extractor", "enricher")
    workflow.add_edge("enricher", "aligner")
    workflow.add_edge("aligner", "pruner")
    workflow.add_edge("pruner", "assembler")
    workflow.add_edge("assembler", "auditor")
    workflow.add_edge("auditor", END)

    app = workflow.compile()

    initial_state = TeamProjectState(
        user_request=user_request,
        research_report=None, skeleton_json=None, enriched_json=None,
        aligned_json=None, abstracted_json=None, assembled_ir=None, final_ir=None,
        last_agent_called=None
    )

    return app.invoke(initial_state)


# ==========================================
# 5. Streamlit UI 布局
# ==========================================

# --- Sidebar ---
st.sidebar.title("Configuration")
modelscope_key = st.sidebar.text_input("ModelScope API Key", type="password")
tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
# classifier_path = st.sidebar.text_input("Local Classifier Path", value="final_activity_classifier_deberta_PMo")
# 这里的 ID 必须换成你自己的！
classifier_path = st.sidebar.text_input("Classifier Path", value="CDBRON/process-model-classifier")

st.sidebar.markdown("---")
st.sidebar.info("Ensure the local classifier folder exists in the same directory as this script.")

# --- Main Area ---
st.title("🤖 Intelligent BPMN Process Generator")
st.markdown(
    "Enter a high-level process description, and the multi-agent system will research, design, and validate a detailed BPMN structure.")

user_input = st.text_area("Process Description", height=150,
                          value="Please design an order to cash process that only includes the roles of 'customer', 'warehouse', 'production', and 'sales'. The key activities are 'order processing', 'production', and 'quality checks', and there should be decision points regarding 'parts availability' and 'production completion'.")

run_btn = st.button("🚀 Generate Process", type="primary")

# --- Execution ---
if run_btn:
    if not modelscope_key or not tavily_key:
        st.error("Please provide both API Keys in the sidebar.")
    else:
        # 创建日志显示区域
        log_expander = st.expander("Execution Logs (Real-time)", expanded=True)
        log_container = log_expander.empty()

        # 捕获输出并运行
        capture = StreamlitCapture(log_container)

        with contextlib.redirect_stdout(capture):
            try:
                with st.spinner("Agents are working... This may take a few minutes."):
                    final_state = run_workflow_logic(user_input, modelscope_key, tavily_key, classifier_path)

                st.success("Workflow Completed Successfully!")

                # 显示结果
                final_ir = final_state.get("final_ir", {})
                st.subheader("Final BPMN IR (JSON)")
                st.json(final_ir)

                # --- 2. 生成 BPMN XML 和 SVG ---
                if "error" not in final_ir and final_ir:
                    st.subheader("2. Generated BPMN Diagram")

                    # 转换步骤
                    try:
                        # A. JSON -> XML
                        converter = BpmnConverter(final_ir)
                        raw_xml = converter.convert()

                        # B. Clean XML (Remove Role Prefix)
                        clean_xml = remove_role_prefix_from_bpmn(raw_xml)

                        # C. XML -> SVG
                        # 使用临时文件名
                        svg_filename = "temp_bpmn_process"
                        svg_path = bpmn_to_svg(clean_xml, svg_filename)

                        if svg_path and os.path.exists(svg_path):
                            # 展示图片
                            st.image(svg_path, caption="Generated BPMN Process", use_container_width=True)

                            # 读取文件内容用于下载
                            with open(svg_path, "rb") as f:
                                svg_bytes = f.read()

                            st.download_button(
                                label="Download SVG Diagram",
                                data=svg_bytes,
                                file_name="bpmn_process.svg",
                                mime="image/svg+xml"
                            )
                        else:
                            st.error("Failed to generate SVG. Please check if Graphviz is installed correctly.")

                        # 下载 XML
                        st.download_button(
                            label="Download BPMN XML",
                            data=clean_xml,
                            file_name="bpmn_process.xml",
                            mime="application/xml"
                        )

                    except Exception as e:
                        st.error(f"Error during BPMN conversion/rendering: {e}")
                        import traceback

                        st.code(traceback.format_exc())

                # 下载按钮
                json_str = json.dumps(final_ir, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="bpmn_process.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"An error occurred during execution: {str(e)}")
                import traceback

                st.code(traceback.format_exc())
