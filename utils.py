import streamlit as st
import json

def format_sources(contexts):
    result = {}
    for doc in contexts:
        source = json.loads(doc["_node_content"])["metadata"]["file_name"] 
        page = json.loads(doc["_node_content"])["metadata"]["page_label"] 
        if source not in result:
            result[source] = set()
        result[source].add(page)
    return result
