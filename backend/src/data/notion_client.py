"""
Notion API client for fetching JKN claims data
"""
import requests
import json
from typing import Dict, List, Any, Optional
from ..utils.config import settings


class NotionClient:
    """Client for interacting with Notion API"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or settings.notion_token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

    def query_database(self, database_id: str) -> Dict[str, Any]:
        """Query a Notion database"""
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        response = requests.post(url, headers=self.headers, json={})
        response.raise_for_status()
        return response.json()

    def get_page_blocks(self, page_id: str) -> Dict[str, Any]:
        """Get blocks from a Notion page"""
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def extract_claim_data(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """Extract claim data from a Notion page"""
        properties = page.get("properties", {})
        extracted = {}

        # Extract common fields
        for key, value in properties.items():
            prop_type = value.get("type")

            if prop_type == "date":
                date_val = value.get("date")
                extracted[key] = date_val.get("start") if date_val else None

            elif prop_type == "number":
                extracted[key] = value.get("number")

            elif prop_type == "rich_text":
                rich_text = value.get("rich_text", [])
                extracted[key] = " ".join([rt.get("text", {}).get("content", "") for rt in rich_text])

            elif prop_type == "title":
                title = value.get("title", [])
                extracted[key] = " ".join([t.get("text", {}).get("content", "") for t in title])

            elif prop_type == "select":
                select = value.get("select")
                extracted[key] = select.get("name") if select else None

            elif prop_type == "url":
                extracted[key] = value.get("url")

        extracted["id"] = page.get("id")
        extracted["created_time"] = page.get("created_time")
        extracted["last_edited_time"] = page.get("last_edited_time")

        return extracted

    def fetch_claims_dataset(self) -> List[Dict[str, Any]]:
        """Fetch all claims from the claims database"""
        response = self.query_database(settings.notion_claims_db_id)
        results = response.get("results", [])

        claims = [self.extract_claim_data(page) for page in results]
        return claims

    def fetch_research_dataset(self) -> List[Dict[str, Any]]:
        """Fetch research dataset"""
        response = self.query_database(settings.notion_dataset_db_id)
        results = response.get("results", [])

        research = [self.extract_claim_data(page) for page in results]
        return research
