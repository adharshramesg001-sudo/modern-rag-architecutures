"""The shared demo knowledge base every architecture retrieves against.

Everything here is small on purpose (this is a demo app, not a production
index) but it is real content: real sentences, real numbers, and a real
noisy/irrelevant document mixed in — so retrieval scores and rankings are
computed from actual text overlap, not scripted.
"""

from __future__ import annotations

from typing import List, Tuple, TypedDict


class Document(TypedDict):
    id: str
    title: str
    text: str


DOCUMENTS: List[Document] = [
    {
        "id": "policy-refund",
        "title": "Refund Policy",
        "text": (
            "Customers can request a refund within 30 days of purchase if the "
            "product is unused and in its original packaging. Refunds are "
            "processed to the original payment method within 5-7 business days. "
            "Digital products and gift cards are non-refundable."
        ),
    },
    {
        "id": "policy-shipping",
        "title": "Shipping Policy",
        "text": (
            "Standard shipping takes 3-5 business days within the country and "
            "7-14 business days internationally. Expedited shipping is available "
            "at checkout for an additional fee. Orders ship from our fulfillment "
            "center in Ohio."
        ),
    },
    {
        "id": "troubleshoot-e104",
        "title": "Troubleshooting Guide: Error E104",
        "text": (
            "Error E104 indicates a network authentication failure between the "
            "device and the cloud gateway. Restart the device, confirm the Wi-Fi "
            "password is correct, and ensure firmware is updated to version 4.2 "
            "or later. If error E104 persists after these steps, replace the "
            "network adapter."
        ),
    },
    {
        "id": "troubleshoot-e210",
        "title": "Troubleshooting Guide: Error E210",
        "text": (
            "Error E210 signals a sensor calibration mismatch. Run the built-in "
            "calibration wizard from Settings, then Diagnostics. If E210 "
            "reappears, the sensor unit may need physical replacement under "
            "warranty."
        ),
    },
    {
        "id": "graph-acme-recall",
        "title": "Acme Corp Product Recall Notice",
        "text": (
            "Acme Corp issued a voluntary recall of Product X after the FDA "
            "identified a contamination risk linked to a component sourced from "
            "Supplier Z. The recall notice was filed with the FDA on August 12."
        ),
    },
    {
        "id": "graph-supplier-z",
        "title": "Supplier Z Overview",
        "text": (
            "Supplier Z manufactures precision components for Acme Corp's "
            "Product X line and also supplies Factory Y, which produces "
            "components for Project Falcon and Project Nova."
        ),
    },
    {
        "id": "graph-projects",
        "title": "Project Falcon and Project Nova",
        "text": (
            "Project Falcon and Project Nova both depend on components "
            "manufactured at Factory Y using parts sourced from Supplier Z, "
            "making them exposed to any disruption in Supplier Z's supply chain."
        ),
    },
    {
        "id": "finance-q3",
        "title": "Q3 Internal Financial Summary",
        "text": (
            "Q3 revenue reached $18.4 million, up 6% quarter-over-quarter, "
            "driven by strong renewal rates in the enterprise segment. Gross "
            "margin held steady at 71%."
        ),
    },
    {
        "id": "finance-benchmark",
        "title": "Industry Benchmark Report Summary (cached)",
        "text": (
            "The latest industry benchmark report shows median SaaS revenue "
            "growth of 9% quarter-over-quarter for companies of our sector and "
            "size in Q3, with top-quartile performers exceeding 15%."
        ),
    },
    {
        "id": "medical-drugx",
        "title": "Drug X Prescribing Information",
        "text": (
            "Drug X at a 10mg dose commonly causes mild nausea and headache in "
            "the first week of use. At 20mg, dizziness and elevated heart rate "
            "have been reported in clinical trials. Patients should consult a "
            "physician before combining Drug X with other medications."
        ),
    },
    {
        "id": "medical-unrelated",
        "title": "Cafeteria Menu Update",
        "text": (
            "The employee cafeteria will introduce a new salad bar and rotating "
            "soup station starting next Monday. Vegetarian and gluten-free "
            "options will be clearly labeled."
        ),
    },
    {
        "id": "multimodal-revenue-context",
        "title": "Quarterly Revenue Report Narrative",
        "text": (
            "This report accompanies the attached revenue chart and table, "
            "covering Q1 2025 through Q4 2025 quarterly revenue in millions of "
            "dollars, highlighting the growth trajectory across the year."
        ),
    },
]

# (subject, relation, object) triples used to build the real networkx
# knowledge graph for Graph RAG and Agentic RAG.
GRAPH_TRIPLES: List[Tuple[str, str, str]] = [
    ("Acme Corp", "manufactures", "Product X"),
    ("Product X", "uses component from", "Supplier Z"),
    ("Acme Corp", "filed recall with", "FDA"),
    ("FDA", "issued", "Recall Notice"),
    ("Recall Notice", "concerns", "Product X"),
    ("Supplier Z", "supplies", "Factory Y"),
    ("Factory Y", "produces parts for", "Project Falcon"),
    ("Factory Y", "produces parts for", "Project Nova"),
    ("Project Falcon", "depends on", "Supplier Z"),
    ("Project Nova", "depends on", "Supplier Z"),
]

# Quarterly revenue used to build the real chart + table for Multimodal RAG,
# consistent with the "$18.4 million" Q3 figure in finance-q3 above.
REVENUE_QUARTERS: List[str] = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
REVENUE_VALUES_M: List[float] = [15.2, 16.8, 18.4, 21.0]
